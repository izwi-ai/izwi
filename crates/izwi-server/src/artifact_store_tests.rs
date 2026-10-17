use super::*;
use crate::{
    batch_runtime::{
        store::{NewJobStage, NewRuntimeJob, QuotaLimits},
        types::{RuntimeJobKind, RuntimeJobStatus, RuntimeStageStatus},
    },
    db::StoreDatabase,
    persistence::LocalMediaStorageProvider,
    speech_history_store::SpeechRouteKind,
    storage_layout,
};
use izwi_hooks::{HookResult, MediaObjectMetadata, StoredMediaObject, StoredMediaStream};
use sea_orm::ConnectionTrait;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
        Arc as StdArc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tempfile::TempDir;
use tokio::sync::Notify;

fn conformance_store(root: &TempDir, provider: Arc<dyn MediaStorageProvider>) -> ArtifactStore {
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(root.path().join("artifacts.sqlite3")),
    ));
    ArtifactStore::new(
        metadata,
        provider,
        ArtifactStoreLimits {
            max_object_bytes: 1024,
            ..Default::default()
        },
    )
    .expect("artifact store")
}

async fn insert_opaque_speech_history_relation(
    db: &StoreDatabase,
    record_id: &str,
    tenant: &ArtifactTenant,
    descriptor: &ArtifactDescriptor,
) {
    use sea_orm::ConnectionTrait;

    let connection = db.connection().await.unwrap();
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                r#"
                    INSERT INTO speech_history_records (
                        id, created_at, route_kind, processing_status, model_id,
                        input_text, generation_time_ms, audio_mime_type,
                        audio_filename, audio_storage_path, audio_media_asset_id,
                        audio_artifact_tenant
                    ) VALUES (?1, 1, 'text_to_speech', 'ready', 'FishAudio-S2-Pro',
                              'fixture', 1.0, ?2, ?3, '', ?4, ?5)
                    "#,
                vec![
                    record_id.into(),
                    descriptor.content_type.clone().into(),
                    descriptor.filename.clone().into(),
                    descriptor.id.as_str().into(),
                    tenant.as_str().into(),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
}

async fn put_test_speech_artifact(
    store: &ArtifactStore,
    tenant: &ArtifactTenant,
    label: &str,
) -> ArtifactDescriptor {
    store
        .put(
            tenant,
            ArtifactWrite {
                content_type: "audio/wav".to_string(),
                filename: Some(format!("{label}.wav")),
                bytes: label.as_bytes().to_vec(),
                retention: ArtifactRetention::Durable,
            },
        )
        .await
        .unwrap()
}

async fn active_tts_lease(
    metadata: &BatchRuntimeStore,
    tenant_key: Option<[u8; 32]>,
) -> StageLease {
    let job = metadata
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("FishAudio-S2-Pro".into()),
            capability: Some("tts".into()),
            route_record_kind: None,
            route_record_id: None,
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: serde_json::json!({"tenant_key": tenant_key}),
            model_snapshot_json: serde_json::json!({"version": 1}),
            retry_policy_json: serde_json::json!({}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .unwrap();
    metadata
        .create_stage(NewJobStage {
            job_id: job.id,
            sequence: 0,
            stage_kind: "tts_synthesize".into(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".into()),
            model_id: Some("FishAudio-S2-Pro".into()),
            max_attempts: 1,
            input_artifact_ids: vec![],
        })
        .await
        .unwrap();
    metadata
        .claim_next_stage("artifact-worker", 60_000)
        .await
        .unwrap()
        .unwrap()
        .lease()
        .unwrap()
}

async fn active_tts_speech_lease(
    metadata: &BatchRuntimeStore,
    tenant_key: Option<[u8; 32]>,
    record_id: &str,
) -> StageLease {
    let job = metadata
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("FishAudio-S2-Pro".into()),
            capability: Some("tts".into()),
            route_record_kind: Some(SpeechRouteKind::TextToSpeech.as_db_value().into()),
            route_record_id: Some(record_id.to_string()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: serde_json::json!({"tenant_key": tenant_key}),
            model_snapshot_json: serde_json::json!({"version": 1}),
            retry_policy_json: serde_json::json!({}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .unwrap();
    metadata
        .create_stage(NewJobStage {
            job_id: job.id,
            sequence: 0,
            stage_kind: "tts_synthesize".into(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".into()),
            model_id: Some("FishAudio-S2-Pro".into()),
            max_attempts: 1,
            input_artifact_ids: vec![],
        })
        .await
        .unwrap();
    let claimed = metadata
        .claim_next_stage("artifact-worker", 60_000)
        .await
        .unwrap()
        .unwrap();
    let lease = claimed.lease().unwrap();
    use sea_orm::ConnectionTrait;
    let connection = metadata.connection().await.unwrap();
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                r#"
                    INSERT INTO speech_history_records (
                        id, created_at, route_kind, processing_status, model_id,
                        input_text, generation_time_ms, audio_mime_type,
                        audio_filename, audio_storage_path, audio_media_asset_id,
                        audio_artifact_tenant, runtime_stage_id, runtime_attempt_token
                    ) VALUES (?1, 1, 'text_to_speech', 'processing', 'FishAudio-S2-Pro',
                              'fixture', 0.0, 'audio/wav', 'speech.wav', '', NULL,
                              NULL, ?2, ?3)
                    "#,
                vec![
                    record_id.into(),
                    lease.stage_id.clone().into(),
                    lease.attempt_token.clone().into(),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    lease
}

fn pcm_attempt_write(bytes: Vec<u8>) -> AttemptArtifactWrite {
    AttemptArtifactWrite {
        publication_key: "speech-pcm/00000000000000000000".into(),
        artifact_kind: RuntimeArtifactKind::Audio,
        artifact_role: RuntimeArtifactRole::OutputIntermediate,
        metadata_json: serde_json::json!({
            "version": 1,
            "sequence": 0,
            "segment": 0,
            "sample_offset": 0,
            "sample_count": bytes.len() / 4,
            "sample_rate": 44_100,
        }),
        runtime_retention_policy: "speech_job_replay".into(),
        object: ArtifactWrite {
            content_type: "audio/pcm-f32le".into(),
            filename: Some("chunk.f32le".into()),
            bytes,
            retention: ArtifactRetention::Job,
        },
    }
}

fn file_attempt_write(path: PathBuf, bytes: &[u8]) -> AttemptArtifactFileWrite {
    AttemptArtifactFileWrite {
        publication_key: "primary-audio".into(),
        artifact_kind: RuntimeArtifactKind::Audio,
        artifact_role: RuntimeArtifactRole::OutputPrimary,
        metadata_json: serde_json::json!({
            "sample_rate": 44_100,
            "sample_count": bytes.len() / 2,
        }),
        runtime_retention_policy: "speech_history".into(),
        object: ArtifactFileWrite {
            content_type: "audio/wav".into(),
            filename: Some("speech.wav".into()),
            path,
            size_bytes: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            retention: ArtifactRetention::Durable,
        },
    }
}

fn test_provider_write_input(
    tenant_scope: &str,
    storage_namespace: &str,
    content_type: &str,
    filename: Option<&str>,
    bytes: &[u8],
    lifetime_ms: u64,
) -> NewProviderWriteReservation {
    let write_id = new_uuid();
    let tenant = ArtifactTenant::parse(tenant_scope).unwrap();
    NewProviderWriteReservation {
        write_id: write_id.clone(),
        tenant_scope: tenant_scope.to_string(),
        storage_namespace: storage_namespace.to_string(),
        content_type: content_type.to_string(),
        filename: filename.map(str::to_string),
        expected_size_bytes: bytes.len() as u64,
        expected_sha256: sha256_hex(bytes),
        lifetime_ms,
        provider_request: MediaWriteRequest {
            namespace: MediaNamespace::Other(storage_namespace.to_string()),
            record_id: write_id,
            preferred_filename: filename.map(str::to_string),
            content_type: content_type.to_string(),
            metadata: tenant_metadata(&tenant),
        },
    }
}

async fn assert_provider_conformance(root: &TempDir, provider: Arc<dyn MediaStorageProvider>) {
    let store = conformance_store(root, provider);
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let other = ArtifactTenant::parse("tenant-b").unwrap();
    let descriptor = store
        .put(
            &tenant,
            ArtifactWrite {
                content_type: "audio/wav".to_string(),
                filename: Some("sample.wav".to_string()),
                bytes: b"bounded artifact".to_vec(),
                retention: ArtifactRetention::Job,
            },
        )
        .await
        .expect("write artifact");

    assert_eq!(descriptor.size_bytes, 16);
    assert_eq!(descriptor.retention, ArtifactRetention::Job);
    assert!(matches!(
        store.read(&other, &descriptor.id).await,
        Err(ArtifactStoreError::NotFound)
    ));
    let read = store
        .read(&tenant, &descriptor.id)
        .await
        .expect("read artifact");
    assert_eq!(read.bytes, b"bounded artifact");
    assert_eq!(read.descriptor, descriptor);

    use tokio::io::AsyncReadExt;
    let mut stream = store
        .read_stream(&tenant, &descriptor.id)
        .await
        .expect("stream artifact");
    let mut streamed_bytes = Vec::new();
    stream
        .reader
        .read_to_end(&mut streamed_bytes)
        .await
        .expect("stream to end");
    assert_eq!(streamed_bytes, b"bounded artifact");

    let public_json = serde_json::to_string(&descriptor).unwrap();
    assert!(!public_json.contains("generated/"));
    assert!(!public_json.contains("private/object"));

    assert!(store.delete(&tenant, &descriptor.id).await.unwrap());
    assert!(matches!(
        store.read(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::NotFound)
    ));
    assert!(!store.delete(&tenant, &descriptor.id).await.unwrap());
}

#[tokio::test]
async fn local_filesystem_provider_conforms_to_opaque_artifact_contract() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(LocalMediaStorageProvider::new(root.path().join("media")));
    assert_provider_conformance(&root, provider.clone()).await;

    let store = conformance_store(&root, provider);
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    for (content_type, filename) in [
        ("image/png", "sample.png"),
        ("video/mp4", "sample.mp4"),
        ("audio/mpeg", "sample.mp3"),
        ("audio/pcm-f32le", "sample.f32le"),
    ] {
        let descriptor = store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: content_type.into(),
                    filename: Some(filename.into()),
                    bytes: b"representative local bytes".to_vec(),
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await
            .unwrap();
        let stored = store.read(&tenant, &descriptor.id).await.unwrap();
        assert_eq!(stored.descriptor.content_type, content_type);
        assert_eq!(stored.bytes, b"representative local bytes");
    }

    assert!(matches!(
        store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: "audio/x-wav".into(),
                    filename: Some("alias.wav".into()),
                    bytes: b"alias".to_vec(),
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await,
        Err(ArtifactStoreError::Provider)
    ));
}

#[tokio::test]
async fn deterministic_remote_like_provider_conforms_to_opaque_artifact_contract() {
    use tokio::io::AsyncReadExt;
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    assert_provider_conformance(&root, provider.clone()).await;

    let store = conformance_store(&root, provider);
    let tenant = ArtifactTenant::parse("tenant-remote").unwrap();
    let descriptor = store
        .put(
            &tenant,
            ArtifactWrite {
                content_type: "audio/wav".into(),
                filename: Some("remote-sample.wav".into()),
                bytes: b"streamable remote artifact payload".to_vec(),
                retention: ArtifactRetention::Durable,
            },
        )
        .await
        .unwrap();
    let mut stream = store.read_stream(&tenant, &descriptor.id).await.unwrap();
    let mut streamed = Vec::new();
    stream.reader.read_to_end(&mut streamed).await.unwrap();
    assert_eq!(streamed, b"streamable remote artifact payload");
    assert_eq!(stream.descriptor.content_type, "audio/wav");
    assert_eq!(stream.descriptor.size_bytes, 34);
}

#[test]
fn scheduling_tenant_mapping_is_stable_and_bounded() {
    assert_eq!(
        ArtifactTenant::from_scheduling_key(None).as_str(),
        "anonymous"
    );
    assert_eq!(
        ArtifactTenant::from_scheduling_key(Some([0xab; 32])).as_str(),
        "abababababababababababababababababababababababababababababababab"
    );
}

#[tokio::test]
async fn attempt_publication_atomically_hides_provider_keys_and_is_idempotent() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("attempt.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant_key = Some([7; 32]);
    let tenant = ArtifactTenant::from_scheduling_key(tenant_key);
    let lease = active_tts_lease(&metadata, tenant_key).await;
    let bytes = [0.1_f32, -0.2, 0.3]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();

    let first = store
        .put_attempt_artifact(
            &tenant,
            &lease,
            pcm_attempt_write(bytes.clone()),
            serde_json::json!({"publication_started": true}),
        )
        .await
        .unwrap();
    let id = ArtifactId::parse(first.media_asset_id.clone().unwrap()).unwrap();
    assert!(first.storage_key.is_none());
    assert_eq!(store.read(&tenant, &id).await.unwrap().bytes, bytes);
    assert!(matches!(
        store
            .read(&ArtifactTenant::parse("other").unwrap(), &id)
            .await,
        Err(ArtifactStoreError::NotFound)
    ));
    let reopened = ArtifactStore::new(
        {
            let mut reopened = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
                root.path().join("attempt.sqlite3"),
            ));
            reopened.set_test_clock(clock.clone());
            Arc::new(reopened)
        },
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    assert_eq!(reopened.read(&tenant, &id).await.unwrap().bytes, bytes);

    let duplicate = store
        .put_attempt_artifact(
            &tenant,
            &lease,
            pcm_attempt_write(bytes),
            serde_json::json!({"publication_started": true}),
        )
        .await
        .unwrap();
    assert_eq!(duplicate.id, first.id);
    assert_eq!(provider.object_count(), 2);
    let mut changed = pcm_attempt_write(0.9_f32.to_le_bytes().to_vec());
    changed.metadata_json["sample_count"] = serde_json::json!(1);
    assert!(matches!(
        store
            .put_attempt_artifact(
                &tenant,
                &lease,
                changed,
                serde_json::json!({"publication_started": true}),
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    assert_eq!(provider.object_count(), 3);
    clock.store(1_100, Ordering::SeqCst);
    provider.set_now_unix_ms(1_100);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 2);
    assert_eq!(provider.object_count(), 1);
}

#[tokio::test]
async fn reserved_file_attempt_publication_is_opaque_and_provider_readable() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(root.path().join("attempt-file.sqlite3")),
    ));
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_object_bytes: 1024,
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let tenant_key = Some([7; 32]);
    let tenant = ArtifactTenant::from_scheduling_key(tenant_key);
    let lease = active_tts_lease(&metadata, tenant_key).await;
    let bytes = vec![0x5a; 128 * 1024 + 3];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let artifact = store
        .put_attempt_file_artifact(
            &tenant,
            &lease,
            file_attempt_write(path, &bytes),
            serde_json::json!({"final_audio_published": true}),
        )
        .await
        .unwrap();
    let id = ArtifactId::parse(artifact.media_asset_id.clone().unwrap()).unwrap();
    assert!(artifact.storage_key.is_none());
    assert_eq!(artifact.size_bytes, Some(bytes.len() as u64));
    assert!(matches!(
        store.read(&tenant, &id).await,
        Err(ArtifactStoreError::TooLarge)
    ));
    let mut verified = store.read_stream(&tenant, &id).await.unwrap();
    assert_eq!(verified.descriptor.size_bytes, bytes.len() as u64);
    let mut empty = [];
    assert_eq!(verified.reader.read(&mut empty).await.unwrap(), 0);
    let mut streamed = Vec::new();
    verified.reader.read_to_end(&mut streamed).await.unwrap();
    assert_eq!(streamed, bytes);
    let asset = metadata
        .get_media_asset(id.as_str())
        .await
        .unwrap()
        .unwrap();
    let stored = provider
        .get(MediaReadRequest {
            key: MediaObjectKey::new(asset.storage_key),
            metadata: tenant_metadata(&tenant),
        })
        .await
        .unwrap();
    assert_eq!(stored.bytes, bytes);

    provider.tamper_single_object(vec![0x2f; bytes.len()]);
    let mut corrupted = store.read_stream(&tenant, &id).await.unwrap();
    let mut discarded = Vec::new();
    assert!(corrupted.reader.read_to_end(&mut discarded).await.is_err());

    provider.tamper_single_object(vec![0x2f; bytes.len() - 1]);
    assert!(matches!(
        store.read_stream(&tenant, &id).await,
        Err(ArtifactStoreError::Integrity("declared size mismatch"))
    ));

    provider.tamper_single_object(vec![0x2f; bytes.len() + 1]);
    assert!(matches!(
        store.read_stream(&tenant, &id).await,
        Err(ArtifactStoreError::Integrity("declared size mismatch"))
    ));
}

#[tokio::test]
async fn speech_final_file_settlement_commits_history_stage_job_and_reservation() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(root.path().join("speech-final.sqlite3")),
    ));
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let record_id = "speech-final-settlement";
    let lease = active_tts_speech_lease(&metadata, None, record_id).await;
    let bytes = vec![0x5a; 4096];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let artifact = store
        .put_attempt_speech_history_file(
            &tenant,
            &lease,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path,
                size_bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
                retention: ArtifactRetention::Durable,
            },
            SpeechHistoryFinalization {
                route_kind: SpeechRouteKind::TextToSpeech,
                record_id: record_id.into(),
                model_id: Some("FishAudio-S2-Pro".into()),
                speaker: Some("alloy".into()),
                language: Some("en".into()),
                saved_voice_id: None,
                speed: Some(1.0),
                input_text: "fixture".into(),
                voice_description: None,
                reference_text: None,
                generation_time_ms: 12.0,
                audio_duration_secs: Some(0.25),
                rtf: Some(0.048),
                tokens_generated: Some(4),
                audio_mime_type: "audio/wav".into(),
                audio_filename: Some("speech.wav".into()),
                artifact_metadata_json: serde_json::json!({
                    "sample_rate": 16_000,
                    "sample_count": 2_000,
                }),
                progress: serde_json::json!({
                    "final_audio_published": true,
                }),
            },
        )
        .await
        .unwrap();

    assert_eq!(
        artifact.publication_key.as_deref(),
        Some("speech-final-audio/v1")
    );
    assert!(artifact.storage_key.is_none());
    assert_eq!(artifact.size_bytes, Some(bytes.len() as u64));
    let media_id = artifact.media_asset_id.clone().unwrap();
    let media = metadata.get_media_asset(&media_id).await.unwrap().unwrap();
    assert_eq!(media.retention_policy, "artifact_durable");
    assert_eq!(media.content_type, "audio/wav");
    assert_eq!(provider.object_count(), 1);

    let stage = metadata.get_stage(&lease.stage_id).await.unwrap().unwrap();
    assert_eq!(stage.status, RuntimeStageStatus::Completed);
    assert_eq!(stage.output_artifact_ids, vec![artifact.id.clone()]);
    assert_eq!(
        stage.progress_json,
        Some(serde_json::json!({"final_audio_published": true}))
    );
    let job = metadata.get_job(&stage.job_id).await.unwrap().unwrap();
    assert_eq!(job.status, RuntimeJobStatus::Completed);
    let follow_up = metadata
        .complete_stage(&lease, vec![artifact.id.clone()])
        .await
        .unwrap()
        .expect("settled stage accepts an exact idempotent completion");
    assert_eq!(follow_up.status, RuntimeStageStatus::Completed);

    use sea_orm::ConnectionTrait;
    let connection = metadata.connection().await.unwrap();
    let history = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                r#"
                    SELECT processing_status, audio_storage_path,
                           audio_media_asset_id, audio_artifact_tenant
                    FROM speech_history_records WHERE id = ?1
                    "#,
                vec![record_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(history.try_get_by_index::<String>(0).unwrap(), "ready");
    assert_eq!(history.try_get_by_index::<String>(1).unwrap(), "");
    assert_eq!(history.try_get_by_index::<String>(2).unwrap(), media_id);
    assert_eq!(
        history.try_get_by_index::<String>(3).unwrap(),
        tenant.as_str()
    );
    let reservation_count = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT COUNT(*) FROM provider_write_reservations WHERE state = 'stored'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(reservation_count, 0);
}

#[tokio::test]
async fn speech_final_file_settlement_fences_cancelled_attempt_and_keeps_object_recoverable() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(root.path().join("speech-final-cancel.sqlite3")),
    ));
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let record_id = "speech-final-cancel";
    let lease = active_tts_speech_lease(&metadata, None, record_id).await;
    let stage = metadata.get_stage(&lease.stage_id).await.unwrap().unwrap();
    metadata
        .cancel_job(&stage.job_id, Some("cancelled before settlement".into()))
        .await
        .unwrap();
    let bytes = vec![0x21; 2048];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let result = store
        .put_attempt_speech_history_file(
            &tenant,
            &lease,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path,
                size_bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
                retention: ArtifactRetention::Durable,
            },
            SpeechHistoryFinalization {
                route_kind: SpeechRouteKind::TextToSpeech,
                record_id: record_id.into(),
                model_id: Some("FishAudio-S2-Pro".into()),
                speaker: None,
                language: None,
                saved_voice_id: None,
                speed: None,
                input_text: "fixture".into(),
                voice_description: None,
                reference_text: None,
                generation_time_ms: 1.0,
                audio_duration_secs: Some(0.1),
                rtf: Some(0.01),
                tokens_generated: Some(1),
                audio_mime_type: "audio/wav".into(),
                audio_filename: Some("speech.wav".into()),
                artifact_metadata_json: serde_json::json!({
                    "sample_rate": 16_000,
                    "sample_count": 800,
                }),
                progress: serde_json::json!({
                    "final_audio_published": true,
                }),
            },
        )
        .await;
    assert!(matches!(result, Err(ArtifactStoreError::Metadata(_))));
    assert_eq!(provider.object_count(), 1);

    use sea_orm::ConnectionTrait;
    let connection = metadata.connection().await.unwrap();
    let state = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT state FROM provider_write_reservations LIMIT 1",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<String>(0)
        .unwrap();
    assert_eq!(state, "cleanup_pending");
    let stage = metadata.get_stage(&lease.stage_id).await.unwrap().unwrap();
    assert_ne!(stage.status, RuntimeStageStatus::Completed);
}

#[tokio::test]
async fn speech_final_file_settlement_fences_expired_attempt_and_keeps_object_recoverable() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(root.path().join("speech-final-expired.sqlite3")),
    ));
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let record_id = "speech-final-expired";
    let lease = active_tts_speech_lease(&metadata, None, record_id).await;

    use sea_orm::ConnectionTrait;
    let connection = metadata.connection().await.unwrap();
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "UPDATE job_stages SET lease_expires_at = 0 WHERE id = ?1",
                vec![lease.stage_id.clone().into()],
            )
            .unwrap(),
        )
        .await
        .unwrap();

    let bytes = vec![0x11; 1024];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();
    let result = store
        .put_attempt_speech_history_file(
            &tenant,
            &lease,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path,
                size_bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
                retention: ArtifactRetention::Durable,
            },
            SpeechHistoryFinalization {
                route_kind: SpeechRouteKind::TextToSpeech,
                record_id: record_id.into(),
                model_id: Some("FishAudio-S2-Pro".into()),
                speaker: None,
                language: None,
                saved_voice_id: None,
                speed: None,
                input_text: "fixture".into(),
                voice_description: None,
                reference_text: None,
                generation_time_ms: 1.0,
                audio_duration_secs: Some(0.1),
                rtf: Some(0.01),
                tokens_generated: Some(1),
                audio_mime_type: "audio/wav".into(),
                audio_filename: Some("speech.wav".into()),
                artifact_metadata_json: serde_json::json!({
                    "sample_rate": 16_000,
                    "sample_count": 800,
                }),
                progress: serde_json::json!({
                    "final_audio_published": true,
                }),
            },
        )
        .await;
    assert!(matches!(result, Err(ArtifactStoreError::Metadata(_))));
    assert_eq!(provider.object_count(), 1);

    let reservation_state = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT state FROM provider_write_reservations LIMIT 1",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<String>(0)
        .unwrap();
    assert_eq!(reservation_state, "cleanup_pending");
    let stage = metadata.get_stage(&lease.stage_id).await.unwrap().unwrap();
    assert_ne!(stage.status, RuntimeStageStatus::Completed);
    let history_status = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT processing_status FROM speech_history_records WHERE id = ?1",
                vec![record_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<String>(0)
        .unwrap();
    assert_eq!(history_status, "processing");
}

#[tokio::test]
async fn speech_final_file_settlement_recovers_after_provider_write_failure() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let db_path = root.path().join("speech-provider-fail.sqlite3");
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let record_id = "speech-provider-fail";
    let lease = active_tts_speech_lease(&metadata, None, record_id).await;

    provider.set_now_unix_ms(u64::MAX);

    let bytes = vec![0x7a; 2048];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let result = store
        .put_attempt_speech_history_file(
            &tenant,
            &lease,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path,
                size_bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
                retention: ArtifactRetention::Durable,
            },
            SpeechHistoryFinalization {
                route_kind: SpeechRouteKind::TextToSpeech,
                record_id: record_id.into(),
                model_id: Some("FishAudio-S2-Pro".into()),
                speaker: None,
                language: None,
                saved_voice_id: None,
                speed: None,
                input_text: "fixture".into(),
                voice_description: None,
                reference_text: None,
                generation_time_ms: 1.0,
                audio_duration_secs: Some(0.1),
                rtf: Some(0.01),
                tokens_generated: Some(1),
                audio_mime_type: "audio/wav".into(),
                audio_filename: Some("speech.wav".into()),
                artifact_metadata_json: serde_json::json!({
                    "sample_rate": 16_000,
                    "sample_count": 800,
                }),
                progress: serde_json::json!({"final_audio_published": true}),
            },
        )
        .await;
    assert!(
        matches!(result, Err(ArtifactStoreError::Provider)),
        "provider failure should return Provider error"
    );
    assert_eq!(
        provider.object_count(),
        0,
        "no provider object should exist after a failed write"
    );

    use sea_orm::ConnectionTrait;
    let connection = metadata.connection().await.unwrap();
    let reservation_count = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT COUNT(*) FROM provider_write_reservations WHERE state = 'cleanup_pending'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(
        reservation_count, 1,
        "abandoned reservation should be cleanup_pending"
    );

    clock.store(1_100, Ordering::SeqCst);
    provider.set_now_unix_ms(1_100);
    let report = store.cleanup_due(64).await.unwrap();
    assert_eq!(
        report.completed, 1,
        "cleanup should delete the abandoned reservation"
    );
    assert_eq!(provider.object_count(), 0);
    let remaining = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT COUNT(*) FROM provider_write_reservations",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(remaining, 0, "all reservations should be cleaned up");
}

#[tokio::test]
async fn speech_final_file_settlement_recovers_orphaned_stored_reservation_after_reopen() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let db_path = root.path().join("speech-crash-reopen.sqlite3");
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata =
        BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let record_id = "speech-crash-reopen";
    let _lease = active_tts_speech_lease(&metadata, None, record_id).await;
    let bytes = vec![0x42; 1024];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let stored = store
        .store_reserved_file(
            &tenant,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path: path.clone(),
                size_bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
                retention: ArtifactRetention::Durable,
            },
        )
        .await
        .unwrap();
    assert_eq!(provider.object_count(), 1, "provider object should exist");

    drop(store);
    drop(metadata);

    let mut reopened_metadata =
        BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path));
    reopened_metadata.set_test_clock(clock.clone());
    let reopened_metadata = Arc::new(reopened_metadata);
    let reopened_store = ArtifactStore::new(
        reopened_metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();

    use sea_orm::ConnectionTrait;
    let connection = reopened_metadata.connection().await.unwrap();
    let state = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT state FROM provider_write_reservations LIMIT 1",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<String>(0)
        .unwrap();
    assert_eq!(
        state, "stored",
        "reservation should still be stored after reopen"
    );

    let history_status = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT processing_status FROM speech_history_records WHERE id = ?1",
                vec![record_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<String>(0)
        .unwrap();
    assert_eq!(
        history_status, "processing",
        "history record should still be processing"
    );

    clock.store(1_100, Ordering::SeqCst);

    provider.set_now_unix_ms(1_100);
    let report = reopened_store.cleanup_due(64).await.unwrap();
    assert_eq!(
        report.completed, 1,
        "cleanup should recover the orphaned stored reservation"
    );
    assert_eq!(
        provider.object_count(),
        0,
        "provider object should be deleted after cleanup"
    );

    let _ = stored;
}

#[tokio::test]
async fn speech_final_file_settlement_replacement_settles_after_stale_attempt_loses_ownership()
{
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("speech-stale-reclaim.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits {
            max_file_object_bytes: 256 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let record_id = "speech-stale-reclaim";

    let job = metadata
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("FishAudio-S2-Pro".into()),
            capability: Some("tts".into()),
            route_record_kind: Some(SpeechRouteKind::TextToSpeech.as_db_value().into()),
            route_record_id: Some(record_id.to_string()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: serde_json::json!({"tenant_key": None::<[u8;32]>}),
            model_snapshot_json: serde_json::json!({"version": 1}),
            retry_policy_json: serde_json::json!({}),
            max_attempts: 2,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .unwrap();
    metadata
        .create_stage(NewJobStage {
            job_id: job.id,
            sequence: 0,
            stage_kind: "tts_synthesize".into(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".into()),
            model_id: Some("FishAudio-S2-Pro".into()),
            max_attempts: 2,
            input_artifact_ids: vec![],
        })
        .await
        .unwrap();
    let first_claimed = metadata
        .claim_next_stage("stale-worker", 60_000)
        .await
        .unwrap()
        .unwrap();
    let first_lease = first_claimed.lease().unwrap();
    {
        use sea_orm::ConnectionTrait;
        let connection = metadata.connection().await.unwrap();
        connection
            .execute_raw(
                crate::db::raw::statement(
                    connection,
                    r#"
                        INSERT INTO speech_history_records (
                            id, created_at, route_kind, processing_status, model_id,
                            input_text, generation_time_ms, audio_mime_type,
                            audio_filename, audio_storage_path, audio_media_asset_id,
                            audio_artifact_tenant, runtime_stage_id, runtime_attempt_token
                        ) VALUES (?1, 1, 'text_to_speech', 'processing', 'FishAudio-S2-Pro',
                                  'fixture', 0.0, 'audio/wav', 'speech.wav', '', NULL,
                                  NULL, ?2, ?3)
                        "#,
                    vec![
                        record_id.into(),
                        first_lease.stage_id.clone().into(),
                        first_lease.attempt_token.clone().into(),
                    ],
                )
                .unwrap(),
            )
            .await
            .unwrap();
    }

    let first_bytes = vec![0x31; 2048];
    let first_path = root.path().join("speech-first.wav");
    tokio::fs::write(&first_path, &first_bytes).await.unwrap();
    let first_stored = store
        .store_reserved_file(
            &tenant,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path: first_path,
                size_bytes: first_bytes.len() as u64,
                sha256: sha256_hex(&first_bytes),
                retention: ArtifactRetention::Durable,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        provider.object_count(),
        1,
        "first provider object should exist"
    );

    metadata
        .fail_stage(
            &first_lease,
            true,
            Some("stale_error".into()),
            Some("stale".into()),
        )
        .await
        .unwrap();

    use sea_orm::ConnectionTrait;
    let connection = metadata.connection().await.unwrap();
    let first_reservation_state = connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT state FROM provider_write_reservations WHERE write_id = ?1",
                vec![first_stored.reservation.write_id.clone().into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<String>(0)
        .unwrap();
    assert_eq!(
        first_reservation_state, "stored",
        "first reservation should still be stored (settlement never attempted)"
    );

    let second_lease = metadata
        .claim_next_stage("replacement-worker", 60_000)
        .await
        .unwrap()
        .unwrap()
        .lease()
        .unwrap();

    {
        use sea_orm::ConnectionTrait;
        let connection = metadata.connection().await.unwrap();
        connection
            .execute_raw(
                crate::db::raw::statement(
                    connection,
                    r#"
                        UPDATE speech_history_records
                        SET runtime_stage_id = ?1, runtime_attempt_token = ?2,
                            processing_status = 'processing'
                        WHERE id = ?3
                        "#,
                    vec![
                        second_lease.stage_id.clone().into(),
                        second_lease.attempt_token.clone().into(),
                        record_id.into(),
                    ],
                )
                .unwrap(),
            )
            .await
            .unwrap();
    }

    let second_bytes = vec![0x52; 4096];
    let second_path = root.path().join("speech-second.wav");
    tokio::fs::write(&second_path, &second_bytes).await.unwrap();
    let artifact = store
        .put_attempt_speech_history_file(
            &tenant,
            &second_lease,
            ArtifactFileWrite {
                content_type: "audio/wav".into(),
                filename: Some("speech.wav".into()),
                path: second_path,
                size_bytes: second_bytes.len() as u64,
                sha256: sha256_hex(&second_bytes),
                retention: ArtifactRetention::Durable,
            },
            SpeechHistoryFinalization {
                route_kind: SpeechRouteKind::TextToSpeech,
                record_id: record_id.into(),
                model_id: Some("FishAudio-S2-Pro".into()),
                speaker: None,
                language: None,
                saved_voice_id: None,
                speed: None,
                input_text: "fixture".into(),
                voice_description: None,
                reference_text: None,
                generation_time_ms: 2.0,
                audio_duration_secs: Some(0.2),
                rtf: Some(0.02),
                tokens_generated: Some(2),
                audio_mime_type: "audio/wav".into(),
                audio_filename: Some("speech.wav".into()),
                artifact_metadata_json: serde_json::json!({
                    "sample_rate": 16_000,
                    "sample_count": 1600,
                }),
                progress: serde_json::json!({"final_audio_published": true}),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        artifact.publication_key.as_deref(),
        Some("speech-final-audio/v1")
    );
    assert_eq!(
        provider.object_count(),
        2,
        "two provider objects: stale (pending cleanup) and replacement"
    );

    let stage = metadata
        .get_stage(&second_lease.stage_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stage.status, RuntimeStageStatus::Completed);

    clock.store(1_100, Ordering::SeqCst);

    provider.set_now_unix_ms(1_100);
    let report = store.cleanup_due(64).await.unwrap();
    assert!(
        report.completed >= 1,
        "cleanup should remove the stale reservation's provider object"
    );
}

#[tokio::test]
async fn verified_stream_reader_handles_empty_reads_and_rejects_short_or_long_bodies() {
    async fn read(bytes: &[u8], expected: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut reader = VerifiedArtifactReader {
            inner: Box::pin(Cursor::new(bytes.to_vec())),
            expected_size: expected.len() as u64,
            expected_digest: sha256_hex(expected),
            total: 0,
            digest: Some(Sha256::new()),
            finished: false,
            scratch: Box::new([0; READ_CHUNK_BYTES]),
        };
        let mut empty = [];
        assert_eq!(reader.read(&mut empty).await?, 0);
        let mut output = Vec::new();
        reader.read_to_end(&mut output).await?;
        Ok(output)
    }

    assert_eq!(read(b"exact", b"exact").await.unwrap(), b"exact");
    assert_eq!(
        read(b"shor", b"short").await.unwrap_err().kind(),
        std::io::ErrorKind::UnexpectedEof
    );
    assert_eq!(
        read(b"longer", b"long").await.unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
}

#[tokio::test]
async fn reserved_file_capability_and_source_identity_fail_before_publication() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("attempt-file-errors.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let lease = active_tts_lease(&metadata, None).await;
    let bytes = b"bounded file".to_vec();
    let path = root.path().join("source.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    provider
        .supports_reserved_files
        .store(false, Ordering::SeqCst);
    let mut unavailable = file_attempt_write(root.path().join("missing.wav"), &bytes);
    unavailable.object.size_bytes = bytes.len() as u64;
    assert!(matches!(
        store
            .put_attempt_file_artifact(&tenant, &lease, unavailable, serde_json::json!({}),)
            .await,
        Err(ArtifactStoreError::ReservedFileWritesUnsupported)
    ));
    assert_eq!(provider.object_count(), 0);

    provider
        .supports_reserved_files
        .store(true, Ordering::SeqCst);
    provider.supports_reserved.store(false, Ordering::SeqCst);
    let unavailable = file_attempt_write(root.path().join("still-missing.wav"), &bytes);
    assert!(matches!(
        store
            .put_attempt_file_artifact(&tenant, &lease, unavailable, serde_json::json!({}),)
            .await,
        Err(ArtifactStoreError::ReservedWritesUnsupported)
    ));
    assert_eq!(provider.object_count(), 0);

    provider.supports_reserved.store(true, Ordering::SeqCst);
    let mut wrong_digest = file_attempt_write(path, &bytes);
    wrong_digest.object.sha256 = "0".repeat(64);
    assert!(matches!(
        store
            .put_attempt_file_artifact(&tenant, &lease, wrong_digest, serde_json::json!({}),)
            .await,
        Err(ArtifactStoreError::Provider)
    ));
    assert_eq!(provider.object_count(), 0);
    clock.store(1_100, Ordering::SeqCst);
    provider.set_now_unix_ms(1_100);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
}

#[tokio::test]
async fn local_reserved_file_restart_cleanup_removes_partial_and_unacknowledged_writes() {
    let root = tempfile::tempdir().unwrap();
    let db_path = root.path().join("reserved-file-recovery.sqlite3");
    let media_root = root.path().join("media");
    let provider = Arc::new(LocalMediaStorageProvider::new(media_root.clone()));
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(db_path.clone()),
    ));
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(100));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let lease = active_tts_lease(&metadata, None).await;
    let bytes = vec![0x35; 128 * 1024 + 3];
    let path = root.path().join("speech.wav");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let mut mismatched = file_attempt_write(path.clone(), &bytes);
    mismatched.object.sha256 = "0".repeat(64);
    assert!(matches!(
        store
            .put_attempt_file_artifact(&tenant, &lease, mismatched, serde_json::json!({}),)
            .await,
        Err(ArtifactStoreError::Provider)
    ));

    let reservation = metadata
        .reserve_provider_write(test_provider_write_input(
            tenant.as_str(),
            "artifact-store",
            "audio/wav",
            Some("speech.wav"),
            &bytes,
            100,
        ))
        .await
        .unwrap();
    let stored = provider
        .put_reserved_file(
            MediaReservedWriteRequest {
                version: MEDIA_RESERVED_WRITE_VERSION,
                write_id: reservation.write_id.clone(),
                expires_at_unix_ms: reservation.expires_at,
                content_length: reservation.expected_size_bytes,
                sha256: reservation.expected_sha256.clone(),
                request: provider_write_request(&reservation),
            },
            path,
        )
        .await
        .unwrap();
    let stored_path = storage_layout::resolve_media_path(&media_root, &stored.key.key).unwrap();
    assert!(stored_path.exists());
    let reserved_root = media_root.join("generated/reserved-writes");
    assert_eq!(std::fs::read_dir(&reserved_root).unwrap().count(), 2);

    drop(store);
    drop(metadata);
    tokio::time::sleep(Duration::from_millis(110)).await;
    let reopened = ArtifactStore::new(
        Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(db_path),
        )),
        provider,
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    assert_eq!(reopened.cleanup_due(64).await.unwrap().completed, 2);
    assert!(!stored_path.exists());
    assert_eq!(std::fs::read_dir(reserved_root).unwrap().count(), 0);
}

#[test]
fn reserved_publication_requires_an_exact_provider_digest() {
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    assert!(matches!(
        validate_stored_write(
            &tenant,
            "audio/wav",
            &sha256_hex(b"audio"),
            5,
            Some(5),
            None,
            Some("tenant-a"),
            "audio/wav",
            1024,
        ),
        Err(ArtifactStoreError::Integrity(
            "provider omitted reserved-write digest"
        ))
    ));
}

#[tokio::test]
async fn stale_attempt_keeps_written_provider_object_recoverable() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("stale-attempt.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let lease = active_tts_lease(&metadata, None).await;
    metadata
        .cancel_job(
            &metadata
                .get_stage(&lease.stage_id)
                .await
                .unwrap()
                .unwrap()
                .job_id,
            Some("test cancellation".into()),
        )
        .await
        .unwrap();

    assert!(matches!(
        store
            .put_attempt_artifact(
                &tenant,
                &lease,
                pcm_attempt_write(0.1_f32.to_le_bytes().to_vec()),
                serde_json::json!({"publication_started": true}),
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    assert_eq!(provider.object_count(), 1);
    clock.store(1_100, Ordering::SeqCst);
    provider.set_now_unix_ms(1_100);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
    assert_eq!(provider.object_count(), 0);
    assert!(metadata
        .speech_pcm_after(
            &metadata
                .get_stage(&lease.stage_id)
                .await
                .unwrap()
                .unwrap()
                .job_id,
            None,
            1,
        )
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn attempt_publication_rejects_a_different_scheduling_tenant() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    // Both clocks are fixed so the 10 ms reservations expire only when the
    // test advances them, however slowly it runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("tenant-fence.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    provider.set_now_unix_ms(1_000);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let lease = active_tts_lease(&metadata, Some([7; 32])).await;
    let wrong_tenant = ArtifactTenant::from_scheduling_key(Some([8; 32]));

    assert!(matches!(
        store
            .put_attempt_artifact(
                &wrong_tenant,
                &lease,
                pcm_attempt_write(0.1_f32.to_le_bytes().to_vec()),
                serde_json::json!({"publication_started": true}),
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    let job_id = metadata
        .get_stage(&lease.stage_id)
        .await
        .unwrap()
        .unwrap()
        .job_id;
    assert!(metadata
        .speech_pcm_after(&job_id, None, 1)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(provider.object_count(), 1);
    clock.store(1_100, Ordering::SeqCst);
    provider.set_now_unix_ms(1_100);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn pcm_entry_limit_rejects_reference_and_recovers_provider_write() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("pcm-limit.sqlite3"),
    ));
    metadata.set_pcm_replay_entry_limit_for_test(1);
    // Both clocks are fixed so the 10 ms reservations cannot expire
    // before the writes land, however slowly the test runs.
    let clock = Arc::new(AtomicI64::new(1_000));
    metadata.set_test_clock(clock.clone());
    provider.set_now_unix_ms(1_000);
    let metadata = Arc::new(metadata);
    let mut store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    store.set_reservation_lifetime_for_test(Duration::from_millis(10));
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let lease = active_tts_lease(&metadata, None).await;
    store
        .put_attempt_artifact(
            &tenant,
            &lease,
            pcm_attempt_write(0.1_f32.to_le_bytes().to_vec()),
            serde_json::json!({"publication_started": true}),
        )
        .await
        .unwrap();
    let mut second = pcm_attempt_write(0.2_f32.to_le_bytes().to_vec());
    second.publication_key = "speech-pcm/00000000000000000001".into();
    second.metadata_json["sequence"] = serde_json::json!(1);
    let error = store
        .put_attempt_artifact(
            &tenant,
            &lease,
            second,
            serde_json::json!({"publication_started": true}),
        )
        .await
        .unwrap_err();
    let ArtifactStoreError::Metadata(error) = error else {
        panic!("unexpected quota error: {error}");
    };
    let error_text = format!("{error:#}");
    assert!(
        error_text.contains("speech_storage_limit"),
        "unexpected quota error: {error_text}"
    );
    let job_id = metadata
        .get_stage(&lease.stage_id)
        .await
        .unwrap()
        .unwrap()
        .job_id;
    assert_eq!(
        metadata
            .speech_pcm_after(&job_id, None, 64)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(provider.object_count(), 2);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 0);
    clock.store(1_010, Ordering::SeqCst);
    provider.set_now_unix_ms(1_010);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
    assert_eq!(provider.object_count(), 1);
}

#[tokio::test]
async fn read_rejects_provider_bytes_that_do_not_match_the_durable_digest() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = conformance_store(&root, provider.clone());
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let descriptor = store
        .put(
            &tenant,
            ArtifactWrite {
                content_type: "application/octet-stream".to_string(),
                filename: None,
                bytes: b"authentic".to_vec(),
                retention: ArtifactRetention::Ephemeral,
            },
        )
        .await
        .unwrap();
    provider.tamper_single_object(b"corrupt!!".to_vec());

    assert!(matches!(
        store.read(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::Integrity("digest mismatch"))
    ));
}

#[tokio::test]
async fn upload_limit_fails_before_calling_the_provider() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = ArtifactStore::new(
        Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(root.path().join("artifacts.sqlite3")),
        )),
        provider.clone(),
        ArtifactStoreLimits {
            max_object_bytes: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();

    assert!(matches!(
        store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: "audio/wav".to_string(),
                    filename: None,
                    bytes: vec![0; 5],
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await,
        Err(ArtifactStoreError::TooLarge)
    ));
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn provider_without_reserved_writes_is_rejected_before_bytes_are_sent() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.supports_reserved.store(false, Ordering::SeqCst);
    let store = conformance_store(&root, provider.clone());
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();

    assert!(matches!(
        store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: "application/octet-stream".to_string(),
                    filename: None,
                    bytes: b"artifact".to_vec(),
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await,
        Err(ArtifactStoreError::ReservedWritesUnsupported)
    ));
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn invalid_provider_storage_keys_are_recovered_after_the_write_fence() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let mut store = conformance_store(&root, provider.clone());
    store.set_reservation_lifetime_for_test(Duration::from_millis(100));
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();

    for key in [
        String::new(),
        "bad\nkey".to_string(),
        "x".repeat(2 * 1024 + 1),
    ] {
        provider.set_next_key(key);
        assert!(matches!(
            store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: "application/octet-stream".to_string(),
                        filename: None,
                        bytes: b"artifact".to_vec(),
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await,
            Err(ArtifactStoreError::InvalidMetadata(
                "invalid provider storage key"
            ))
        ));
        tokio::time::sleep(Duration::from_millis(110)).await;
        assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
        assert_eq!(provider.object_count(), 0);
    }
}

#[tokio::test]
async fn exact_provider_request_without_returned_key_is_recovered_after_reopen() {
    let root = tempfile::tempdir().unwrap();
    let db_path = root.path().join("artifacts.sqlite3");
    let provider = Arc::new(MemoryMediaProvider::default());
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(db_path.clone()),
    ));
    let write_id = new_uuid();
    let mut request_metadata = HookMetadata::new();
    request_metadata.insert("tenant_id".into(), "tenant-a".into());
    request_metadata.insert("workflow_stage".into(), "speech-finalize".into());
    let reservation = metadata
        .reserve_provider_write(NewProviderWriteReservation {
            write_id: write_id.clone(),
            tenant_scope: "tenant-a".into(),
            storage_namespace: "generated_speech".into(),
            content_type: "application/octet-stream".into(),
            filename: Some("crash.bin".into()),
            expected_size_bytes: 5,
            expected_sha256: sha256_hex(b"crash"),
            lifetime_ms: 100,
            provider_request: MediaWriteRequest {
                namespace: MediaNamespace::GeneratedSpeech,
                record_id: write_id,
                preferred_filename: Some("crash.bin".into()),
                content_type: "application/octet-stream".into(),
                metadata: request_metadata,
            },
        })
        .await
        .unwrap();
    provider
        .put_reserved(
            MediaReservedWriteRequest {
                version: MEDIA_RESERVED_WRITE_VERSION,
                write_id: reservation.write_id.clone(),
                expires_at_unix_ms: reservation.expires_at,
                content_length: reservation.expected_size_bytes,
                sha256: reservation.expected_sha256.clone(),
                request: provider_write_request(&reservation),
            },
            b"crash".to_vec(),
        )
        .await
        .unwrap();
    assert_eq!(provider.object_count(), 1);
    drop(metadata);
    tokio::time::sleep(Duration::from_millis(110)).await;

    let reopened = ArtifactStore::new(
        Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(db_path),
        )),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    assert_eq!(reopened.cleanup_due(64).await.unwrap().completed, 1);
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn delayed_reserved_put_cannot_publish_after_recovery_fences_its_write_id() {
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let write_id = uuid::Uuid::new_v4().to_string();
    let media_request = MediaWriteRequest {
        namespace: MediaNamespace::Other("artifact-store".into()),
        record_id: write_id.clone(),
        preferred_filename: Some("race.bin".into()),
        content_type: "application/octet-stream".into(),
        metadata: HookMetadata::new(),
    };
    let write_request = MediaReservedWriteRequest {
        version: MEDIA_RESERVED_WRITE_VERSION,
        write_id: write_id.clone(),
        expires_at_unix_ms: 1_010,
        content_length: 4,
        sha256: sha256_hex(b"race"),
        request: media_request.clone(),
    };
    let entered = StdArc::new(Notify::new());
    let release = StdArc::new(Notify::new());
    let entered_wait = entered.notified();
    provider.set_reserved_put_gate(entered.clone(), release.clone());
    let delayed_provider = provider.clone();
    let delayed_put = tokio::spawn(async move {
        delayed_provider
            .put_reserved(write_request, b"race".to_vec())
            .await
    });
    entered_wait.await;

    provider.set_now_unix_ms(1_010);
    assert!(matches!(
        provider
            .recover_reserved_write(MediaReservedWriteRecoveryRequest {
                version: MEDIA_RESERVED_WRITE_VERSION,
                write_id,
                expires_at_unix_ms: 1_010,
                content_length: 4,
                sha256: sha256_hex(b"race"),
                request: media_request,
                storage_key: None,
            })
            .await,
        Err(HookError::NotFound(_))
    ));
    // Rewind the deterministic clock so expiry alone cannot make this pass:
    // the recovery fence must reject the delayed publisher.
    provider.set_now_unix_ms(1_000);
    release.notify_one();
    assert!(delayed_put.await.unwrap().is_err());
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn provider_write_capacity_fails_before_bytes_are_sent() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_provider_write_capacity_for_test(1);
    let metadata = Arc::new(metadata);
    let input = test_provider_write_input(
        "tenant-a",
        "artifact-store",
        "application/octet-stream",
        None,
        b"x",
        60_000,
    );
    metadata
        .reserve_provider_write(input.clone())
        .await
        .unwrap();
    let store =
        ArtifactStore::new(metadata, provider.clone(), ArtifactStoreLimits::default()).unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    assert!(matches!(
        store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: input.content_type,
                    filename: input.filename,
                    bytes: b"x".to_vec(),
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn provider_write_cleanup_claim_is_exact_and_nonduplicating() {
    let root = tempfile::tempdir().unwrap();
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let reservation = metadata
        .reserve_provider_write(test_provider_write_input(
            "tenant-a",
            "artifact-store",
            "application/octet-stream",
            None,
            b"x",
            10,
        ))
        .await
        .unwrap();
    metadata
        .record_provider_write_stored(&reservation, "private/object/fenced")
        .await
        .unwrap()
        .unwrap();
    clock.store(1_010, Ordering::SeqCst);
    let first = metadata.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(first.len(), 1);
    assert!(first[0].cleanup_claim_token.is_some());
    assert!(metadata
        .claim_due_provider_write_cleanup(64)
        .await
        .unwrap()
        .is_empty());
    assert!(metadata
        .publish_reserved_opaque_artifact(
            &first[0],
            NewMediaAsset {
                asset_kind: "opaque_artifact".into(),
                storage_namespace: "artifact_store_v1".into(),
                storage_key: "private/object/fenced".into(),
                content_type: "application/octet-stream".into(),
                filename: None,
                size_bytes: 1,
                sha256: Some(sha256_hex(b"x")),
                duration_secs: None,
                sample_rate_hz: None,
                channel_count: None,
                peak_amplitude: None,
                rms_amplitude: None,
                source_asset_id: None,
                canonical_profile_version: None,
                scan_status: "not_required".into(),
                retention_policy: "artifact_ephemeral".into(),
                metadata_json: serde_json::json!({}),
            },
        )
        .await
        .unwrap()
        .is_none());
    let forged = ProviderWriteReservation {
        cleanup_claim_token: Some(uuid::Uuid::new_v4().to_string()),
        ..first[0].clone()
    };
    assert!(!metadata
        .complete_provider_write_cleanup(&forged)
        .await
        .unwrap());
    assert!(metadata
        .complete_provider_write_cleanup(&first[0])
        .await
        .unwrap());
}

#[tokio::test]
async fn filename_cannot_escape_provider_namespace() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = conformance_store(&root, provider.clone());
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();

    for filename in ["../secret", "nested/file.wav", "nested\\file.wav"] {
        assert!(matches!(
            store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: "audio/wav".to_string(),
                        filename: Some(filename.to_string()),
                        bytes: b"artifact".to_vec(),
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await,
            Err(ArtifactStoreError::InvalidInput(
                "invalid artifact filename"
            ))
        ));
    }
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn failed_physical_delete_remains_logically_deleted() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = conformance_store(&root, provider.clone());
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let descriptor = store
        .put(
            &tenant,
            ArtifactWrite {
                content_type: "audio/wav".to_string(),
                filename: None,
                bytes: b"artifact".to_vec(),
                retention: ArtifactRetention::Durable,
            },
        )
        .await
        .unwrap();
    provider.fail_deletes.store(true, Ordering::SeqCst);

    assert!(matches!(
        store.delete(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    assert!(matches!(
        store.read(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::NotFound)
    ));
    assert_eq!(provider.object_count(), 1);
    provider.fail_deletes.store(false, Ordering::SeqCst);
    assert!(!store.delete(&tenant, &descriptor.id).await.unwrap());
    assert_eq!(provider.object_count(), 0);
}

/// DUR-F6: retention labels drive expiry. Ephemeral objects go after
/// their retention, durable ones never, and the tombstone row is purged
/// once the object is gone and the grace passed.
#[tokio::test]
async fn retention_labels_expire_ephemeral_objects_and_keep_durable_ones() {
    let root = tempfile::tempdir().unwrap();
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let ephemeral = put_test_artifact(&store, &tenant, "ephemeral").await;
    let durable = store
        .put(
            &tenant,
            ArtifactWrite {
                content_type: "application/octet-stream".to_string(),
                filename: Some("durable.bin".to_string()),
                bytes: b"durable".to_vec(),
                retention: ArtifactRetention::Durable,
            },
        )
        .await
        .unwrap();
    let policy = RetentionPolicy {
        terminal_job_ms: 10_000,
        ephemeral_artifact_ms: 5_000,
        worker_heartbeat_ms: 10_000,
        media_tombstone_ms: 1_000,
        legacy_idempotency_ms: 10_000,
    };

    assert_eq!(
        store.expire_retained(policy, 64).await.unwrap().inspected,
        0
    );
    clock.store(7_000, Ordering::SeqCst);
    provider.set_now_unix_ms(7_000);
    assert_eq!(
        store.expire_retained(policy, 64).await.unwrap(),
        ArtifactCleanupReport {
            inspected: 1,
            completed: 1,
            deferred: 0,
        }
    );
    assert!(matches!(
        store.stat(&tenant, &ephemeral.id).await,
        Err(ArtifactStoreError::NotFound)
    ));
    assert_eq!(store.stat(&tenant, &durable.id).await.unwrap(), durable);
    assert_eq!(provider.object_count(), 1);

    // The tombstone outlives its object by the grace, then goes.
    assert_eq!(
        metadata
            .sweep_retention(policy, 64)
            .await
            .unwrap()
            .media_tombstones,
        0
    );
    clock.store(9_000, Ordering::SeqCst);
    assert_eq!(
        metadata
            .sweep_retention(policy, 64)
            .await
            .unwrap()
            .media_tombstones,
        1
    );
    assert!(metadata
        .get_media_asset(ephemeral.id.as_str())
        .await
        .unwrap()
        .is_none());
    assert!(metadata
        .get_media_asset(durable.id.as_str())
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn cleanup_intent_survives_store_reopen_and_retries_known_dead_object() {
    let root = tempfile::tempdir().unwrap();
    let db_path = root.path().join("artifacts.sqlite3");
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
    let mut metadata =
        BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
    metadata.set_test_clock(clock.clone());
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let store = ArtifactStore::new(
        Arc::new(metadata),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let descriptor = put_test_artifact(&store, &tenant, "reopen").await;
    provider.fail_deletes.store(true, Ordering::SeqCst);
    assert!(matches!(
        store.delete(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    drop(store);

    clock.store(2_000, Ordering::SeqCst);
    let mut reopened = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path));
    reopened.set_test_clock(clock);
    provider.fail_deletes.store(false, Ordering::SeqCst);
    let reopened = ArtifactStore::new(
        Arc::new(reopened),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    assert_eq!(
        reopened.cleanup_due(64).await.unwrap(),
        ArtifactCleanupReport {
            inspected: 1,
            completed: 1,
            deferred: 0,
        }
    );
    assert_eq!(provider.object_count(), 0);
    assert!(reopened
        .metadata
        .due_artifact_cleanup_intents(64)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn opaque_speech_delete_atomically_hides_row_and_retries_provider_cleanup() {
    use sea_orm::ConnectionTrait;

    let root = tempfile::tempdir().unwrap();
    let db = StoreDatabase::new(root.path().join("speech-delete.sqlite3"));
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(db.clone());
    metadata.set_test_clock(clock.clone());
    let metadata = Arc::new(metadata);
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-speech-delete").unwrap();
    let descriptor = put_test_speech_artifact(&store, &tenant, "history").await;
    insert_opaque_speech_history_relation(&db, "speech-delete", &tenant, &descriptor).await;
    assert!(matches!(
        store.delete(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    assert_eq!(
        store.stat(&tenant, &descriptor.id).await.unwrap(),
        descriptor
    );
    provider.fail_deletes.store(true, Ordering::SeqCst);

    assert!(
        store
            .delete_speech_history_audio(
                "text_to_speech",
                "speech-delete",
                &tenant,
                &descriptor.id,
            )
            .await
            .unwrap()
    );
    let connection = db.connection().await.unwrap();
    assert!(connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT 1 FROM speech_history_records WHERE id = 'speech-delete'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        store.stat(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::NotFound)
    ));
    assert_eq!(provider.object_count(), 1);
    drop(store);

    clock.store(2_000, Ordering::SeqCst);
    provider.fail_deletes.store(false, Ordering::SeqCst);
    let mut reopened_metadata = BatchRuntimeStore::initialize_with_database(db);
    reopened_metadata.set_test_clock(clock);
    let reopened = ArtifactStore::new(
        Arc::new(reopened_metadata),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    assert_eq!(reopened.cleanup_due(64).await.unwrap().completed, 1);
    assert_eq!(provider.object_count(), 0);
}

#[tokio::test]
async fn opaque_speech_delete_capacity_failure_rolls_back_row_and_tombstone() {
    use sea_orm::ConnectionTrait;

    let root = tempfile::tempdir().unwrap();
    let db = StoreDatabase::new(root.path().join("speech-delete-capacity.sqlite3"));
    let mut metadata = BatchRuntimeStore::initialize_with_database(db.clone());
    metadata.set_artifact_cleanup_capacity_for_test(0);
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = ArtifactStore::new(
        Arc::new(metadata),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-speech-capacity").unwrap();
    let descriptor = put_test_speech_artifact(&store, &tenant, "history").await;
    insert_opaque_speech_history_relation(&db, "speech-capacity", &tenant, &descriptor).await;

    assert!(matches!(
        store
            .delete_speech_history_audio(
                "text_to_speech",
                "speech-capacity",
                &tenant,
                &descriptor.id,
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    assert_eq!(
        store.stat(&tenant, &descriptor.id).await.unwrap(),
        descriptor
    );
    let connection = db.connection().await.unwrap();
    assert!(connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT 1 FROM speech_history_records WHERE id = 'speech-capacity'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .is_some());
    assert_eq!(provider.object_count(), 1);
}

#[tokio::test]
async fn opaque_speech_delete_rejects_mismatched_authoritative_relation() {
    use sea_orm::ConnectionTrait;

    let root = tempfile::tempdir().unwrap();
    let db = StoreDatabase::new(root.path().join("speech-delete-mismatch.sqlite3"));
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(db.clone()));
    let provider = Arc::new(MemoryMediaProvider::default());
    let store =
        ArtifactStore::new(metadata.clone(), provider.clone(), Default::default()).unwrap();
    let tenant = ArtifactTenant::parse("tenant-speech-mismatch").unwrap();
    let descriptor = put_test_speech_artifact(&store, &tenant, "history").await;
    insert_opaque_speech_history_relation(&db, "speech-mismatch", &tenant, &descriptor).await;
    let connection = db.connection().await.unwrap();

    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "UPDATE speech_history_records SET processing_status = 'processing' WHERE id = 'speech-mismatch'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(!store
        .delete_speech_history_audio(
            "text_to_speech",
            "speech-mismatch",
            &tenant,
            &descriptor.id,
        )
        .await
        .unwrap());
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "UPDATE speech_history_records SET processing_status = 'ready', audio_mime_type = 'audio/mpeg' WHERE id = 'speech-mismatch'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(matches!(
        store
            .delete_speech_history_audio(
                "text_to_speech",
                "speech-mismatch",
                &tenant,
                &descriptor.id,
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "UPDATE speech_history_records SET audio_mime_type = 'audio/wav' WHERE id = 'speech-mismatch'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "UPDATE media_assets SET retention_policy = 'artifact_job' WHERE id = ?1",
                vec![descriptor.id.as_str().into()],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(matches!(
        store
            .delete_speech_history_audio(
                "text_to_speech",
                "speech-mismatch",
                &tenant,
                &descriptor.id,
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "UPDATE media_assets SET retention_policy = 'artifact_durable', metadata_json = ?1 WHERE id = ?2",
                vec![
                    serde_json::json!({
                        "artifact_store": {"version": 1, "tenant_id": "other-tenant"}
                    })
                    .to_string()
                    .into(),
                    descriptor.id.as_str().into(),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(metadata
        .delete_opaque_speech_history_record(
            "text_to_speech",
            "speech-mismatch",
            descriptor.id.as_str(),
            tenant.as_str(),
        )
        .await
        .is_err());

    assert!(connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT 1 FROM speech_history_records WHERE id = 'speech-mismatch'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .is_some());
    assert_eq!(provider.object_count(), 1);
}

#[tokio::test]
async fn opaque_speech_delete_acknowledges_post_commit_cleanup_bookkeeping_failure() {
    use sea_orm::ConnectionTrait;

    let root = tempfile::tempdir().unwrap();
    let db = StoreDatabase::new(root.path().join("speech-delete-bookkeeping.sqlite3"));
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(db.clone()));
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = ArtifactStore::new(metadata, provider.clone(), Default::default()).unwrap();
    let tenant = ArtifactTenant::parse("tenant-speech-bookkeeping").unwrap();
    let descriptor = put_test_speech_artifact(&store, &tenant, "history").await;
    insert_opaque_speech_history_relation(&db, "speech-bookkeeping", &tenant, &descriptor)
        .await;
    let connection = db.connection().await.unwrap();
    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                r#"
                    CREATE TRIGGER reject_cleanup_intent_delete
                    BEFORE DELETE ON artifact_cleanup_intents
                    BEGIN
                        SELECT RAISE(FAIL, 'injected cleanup bookkeeping failure');
                    END
                    "#,
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap();

    assert!(store
        .delete_speech_history_audio(
            "text_to_speech",
            "speech-bookkeeping",
            &tenant,
            &descriptor.id,
        )
        .await
        .unwrap());
    assert!(matches!(
        store.stat(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::NotFound)
    ));
    assert_eq!(provider.object_count(), 0);
    assert_eq!(
        store
            .metadata
            .due_artifact_cleanup_intents(64)
            .await
            .unwrap()
            .len(),
        1
    );

    connection
        .execute_raw(
            crate::db::raw::statement(
                connection,
                "DROP TRIGGER reject_cleanup_intent_delete",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
}

#[tokio::test]
async fn opaque_speech_delete_waits_for_runtime_ownership_teardown() {
    use sea_orm::ConnectionTrait;

    let root = tempfile::tempdir().unwrap();
    let db = StoreDatabase::new(root.path().join("speech-delete-owner.sqlite3"));
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(db.clone()));
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = ArtifactStore::new(metadata.clone(), provider, Default::default()).unwrap();
    let tenant = ArtifactTenant::from_scheduling_key(None);
    let descriptor = put_test_speech_artifact(&store, &tenant, "history").await;
    insert_opaque_speech_history_relation(&db, "speech-owner", &tenant, &descriptor).await;
    let job = metadata
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("FishAudio-S2-Pro".into()),
            capability: Some("tts".into()),
            route_record_kind: Some("text_to_speech".into()),
            route_record_id: Some("speech-owner".into()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: serde_json::json!({"tenant_key": null}),
            model_snapshot_json: serde_json::json!({}),
            retry_policy_json: serde_json::json!({}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .unwrap();

    assert!(matches!(
        store
            .delete_speech_history_audio(
                "text_to_speech",
                "speech-owner",
                &tenant,
                &descriptor.id,
            )
            .await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    let connection = db.connection().await.unwrap();
    assert!(connection
        .query_one_raw(
            crate::db::raw::statement(
                connection,
                "SELECT 1 FROM speech_history_records WHERE id = 'speech-owner'",
                vec![],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .is_some());

    metadata.cancel_job(&job.id, None).await.unwrap().unwrap();
    assert!(store
        .delete_speech_history_audio("text_to_speech", "speech-owner", &tenant, &descriptor.id,)
        .await
        .unwrap());
    assert!(metadata.retry_job(&job.id).await.unwrap().is_none());
}

#[tokio::test]
async fn cleanup_capacity_failure_happens_before_second_tombstone() {
    let root = tempfile::tempdir().unwrap();
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_artifact_cleanup_capacity_for_test(1);
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = ArtifactStore::new(
        Arc::new(metadata),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let first = put_test_artifact(&store, &tenant, "first").await;
    let second = put_test_artifact(&store, &tenant, "second").await;
    provider.fail_deletes.store(true, Ordering::SeqCst);
    assert!(matches!(
        store.delete(&tenant, &first.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    assert!(matches!(
        store.delete(&tenant, &second.id).await,
        Err(ArtifactStoreError::Metadata(_))
    ));
    assert_eq!(store.stat(&tenant, &second.id).await.unwrap(), second);
}

/// The cleanup quota rows equal the pending intents, per tenant and in
/// total.
async fn assert_cleanup_quota_matches_intents(metadata: &BatchRuntimeStore) {
    let db = metadata.connection().await.unwrap();
    let rows = db
        .query_all_raw(sea_orm::Statement::from_string(
            db.get_database_backend(),
            "SELECT q.scope, q.used, (SELECT COUNT(*) FROM artifact_cleanup_intents c WHERE q.scope = '' OR c.tenant_scope = q.scope) FROM runtime_quota_usage q WHERE q.quota_kind = 'artifact_cleanup'",
        ))
        .await
        .unwrap();
    assert!(!rows.is_empty());
    for row in rows {
        let scope: String = row.try_get_by_index(0).unwrap();
        let used: i64 = row.try_get_by_index(1).unwrap();
        let pending: i64 = row.try_get_by_index(2).unwrap();
        assert_eq!(used, pending, "cleanup quota of scope {scope:?}");
    }
}

/// DUR-F4 for cleanup intents: pending intents are counted per tenant and
/// in total, a tenant at its cap cannot use up the global cap, and a
/// completed cleanup gives its unit back.
#[tokio::test]
async fn cleanup_quota_is_per_tenant_and_released_on_completion() {
    let root = tempfile::tempdir().unwrap();
    let clock = Arc::new(AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    metadata.set_artifact_cleanup_limits_for_test(QuotaLimits {
        global: 2,
        per_tenant: 1,
    });
    let metadata = Arc::new(metadata);
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant_a = ArtifactTenant::parse("tenant-a").unwrap();
    let tenant_b = ArtifactTenant::parse("tenant-b").unwrap();
    let tenant_c = ArtifactTenant::parse("tenant-c").unwrap();
    let a1 = put_test_artifact(&store, &tenant_a, "a1").await;
    let a2 = put_test_artifact(&store, &tenant_a, "a2").await;
    let b1 = put_test_artifact(&store, &tenant_b, "b1").await;
    let c1 = put_test_artifact(&store, &tenant_c, "c1").await;
    provider.fail_deletes.store(true, Ordering::SeqCst);

    assert!(matches!(
        store.delete(&tenant_a, &a1.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    assert!(
        matches!(
            store.delete(&tenant_a, &a2.id).await,
            Err(ArtifactStoreError::Metadata(_))
        ),
        "tenant-a is at its cap"
    );
    assert!(matches!(
        store.delete(&tenant_b, &b1.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    assert!(
        matches!(
            store.delete(&tenant_c, &c1.id).await,
            Err(ArtifactStoreError::Metadata(_))
        ),
        "the global cap is reached"
    );
    assert_eq!(store.stat(&tenant_a, &a2.id).await.unwrap(), a2);
    assert_cleanup_quota_matches_intents(&metadata).await;

    // A repeated delete shares the pending intent and takes no new unit.
    assert!(matches!(
        store.delete(&tenant_a, &a1.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    assert_cleanup_quota_matches_intents(&metadata).await;

    provider.fail_deletes.store(false, Ordering::SeqCst);
    // Past the longest retry backoff.
    clock.store(1_000 + 60 * 60 * 1000, Ordering::SeqCst);
    assert_eq!(store.cleanup_due(64).await.unwrap().completed, 2);
    assert_cleanup_quota_matches_intents(&metadata).await;
    assert!(store.delete(&tenant_a, &a2.id).await.unwrap());
    assert!(store.delete(&tenant_c, &c1.id).await.unwrap());
    assert_cleanup_quota_matches_intents(&metadata).await;
}

/// Migration 3 seeds the cleanup quota from the intents that exist.
#[tokio::test]
async fn cleanup_quota_is_seeded_from_existing_intents_on_upgrade() {
    let root = tempfile::tempdir().unwrap();
    let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
        StoreDatabase::new(root.path().join("artifacts.sqlite3")),
    ));
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = ArtifactStore::new(
        metadata.clone(),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant_a = ArtifactTenant::parse("tenant-a").unwrap();
    let tenant_b = ArtifactTenant::parse("tenant-b").unwrap();
    provider.fail_deletes.store(true, Ordering::SeqCst);
    for (tenant, label) in [(&tenant_a, "a1"), (&tenant_a, "a2"), (&tenant_b, "b1")] {
        let artifact = put_test_artifact(&store, tenant, label).await;
        assert!(store.delete(tenant, &artifact.id).await.is_err());
    }
    let db = metadata.connection().await.unwrap();
    // A database from before migration 3: intents but no cleanup counts.
    for sql in [
        "DELETE FROM runtime_quota_usage WHERE quota_kind = 'artifact_cleanup'",
        "DELETE FROM schema_migrations WHERE component = 'app' AND version = 3",
    ] {
        db.execute_unprepared(sql).await.unwrap();
    }
    crate::db::migrator::Migrator::up(db).await.unwrap();
    assert_cleanup_quota_matches_intents(&metadata).await;
}

#[tokio::test]
async fn provider_not_found_completes_cleanup_idempotently() {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(MemoryMediaProvider::default());
    let store = conformance_store(&root, provider.clone());
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let descriptor = put_test_artifact(&store, &tenant, "missing").await;
    provider.remove_single_object();

    assert!(store.delete(&tenant, &descriptor.id).await.unwrap());
    assert!(store
        .metadata
        .due_artifact_cleanup_intents(64)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn due_cleanup_is_bounded_and_never_selects_active_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let store = ArtifactStore::new(
        Arc::new(metadata),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let active = put_test_artifact(&store, &tenant, "active").await;
    assert_eq!(store.cleanup_due(0).await.unwrap().inspected, 0);
    assert_eq!(store.cleanup_due(usize::MAX).await.unwrap().inspected, 0);
    assert_eq!(store.stat(&tenant, &active.id).await.unwrap(), active);

    provider.fail_deletes.store(true, Ordering::SeqCst);
    for index in 0..65 {
        let artifact = put_test_artifact(&store, &tenant, &format!("dead-{index}")).await;
        assert!(matches!(
            store.delete(&tenant, &artifact.id).await,
            Err(ArtifactStoreError::DeleteIncomplete)
        ));
    }
    clock.store(2_000, Ordering::SeqCst);
    provider.fail_deletes.store(false, Ordering::SeqCst);
    let first = store.cleanup_due(usize::MAX).await.unwrap();
    assert_eq!(first.inspected, 64);
    assert_eq!(first.completed, 64);
    let second = store.cleanup_due(usize::MAX).await.unwrap();
    assert_eq!(second.inspected, 1);
    assert_eq!(second.completed, 1);
    assert_eq!(store.stat(&tenant, &active.id).await.unwrap(), active);
}

#[tokio::test]
async fn duplicate_cleaners_and_retry_metadata_remain_safe_and_bounded() {
    let root = tempfile::tempdir().unwrap();
    let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
    let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("artifacts.sqlite3"),
    ));
    metadata.set_test_clock(clock.clone());
    let provider = Arc::new(MemoryMediaProvider::default());
    provider.set_now_unix_ms(1_000);
    let store = ArtifactStore::new(
        Arc::new(metadata),
        provider.clone(),
        ArtifactStoreLimits::default(),
    )
    .unwrap();
    let tenant = ArtifactTenant::parse("tenant-a").unwrap();
    let descriptor = put_test_artifact(&store, &tenant, "duplicate").await;
    provider.set_delete_error("é".repeat(600));
    assert!(matches!(
        store.delete(&tenant, &descriptor.id).await,
        Err(ArtifactStoreError::DeleteIncomplete)
    ));
    let mut intent = store
        .metadata
        .artifact_cleanup_intent_for_storage_key(
            &store
                .resolve(&tenant, &descriptor.id)
                .await
                .unwrap()
                .unwrap()
                .storage_key,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(intent.attempt_count, 1);
    assert!(intent.last_error.as_ref().unwrap().len() <= 512);
    for _ in 0..14 {
        clock.store(intent.available_at as i64, Ordering::SeqCst);
        assert!(matches!(
            store.delete(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::DeleteIncomplete)
        ));
        let next = store
            .metadata
            .artifact_cleanup_intent_for_storage_key(&intent.storage_key)
            .await
            .unwrap()
            .unwrap();
        assert!(next.available_at - intent.available_at <= 60 * 60 * 1000);
        intent = next;
    }

    provider.clear_delete_error();
    let first_cleaner = intent.clone();
    let second_cleaner = intent;
    assert!(store.cleanup_intent(&first_cleaner).await.unwrap());
    assert!(store.cleanup_intent(&second_cleaner).await.unwrap());
    assert!(store
        .metadata
        .artifact_cleanup_intent_for_storage_key(&first_cleaner.storage_key)
        .await
        .unwrap()
        .is_none());
}

async fn put_test_artifact(
    store: &ArtifactStore,
    tenant: &ArtifactTenant,
    label: &str,
) -> ArtifactDescriptor {
    store
        .put(
            tenant,
            ArtifactWrite {
                content_type: "application/octet-stream".to_string(),
                filename: Some(format!("{label}.bin")),
                bytes: label.as_bytes().to_vec(),
                retention: ArtifactRetention::Ephemeral,
            },
        )
        .await
        .unwrap()
}

#[derive(Clone)]
struct MemoryObject {
    bytes: Vec<u8>,
    metadata: MediaObjectMetadata,
}

struct MemoryReservedWrite {
    request: MediaWriteRequest,
    size_bytes: u64,
    sha256: String,
    key: String,
}

#[derive(Default)]
struct MemoryProviderState {
    objects: BTreeMap<String, MemoryObject>,
    reserved_writes: BTreeMap<String, MemoryReservedWrite>,
    fenced_write_ids: BTreeSet<String>,
}

struct MemoryMediaProvider {
    state: Mutex<MemoryProviderState>,
    supports_reserved: AtomicBool,
    supports_reserved_files: AtomicBool,
    now_unix_ms: AtomicU64,
    reserved_put_gate: Mutex<Option<(StdArc<Notify>, StdArc<Notify>)>>,
    fail_deletes: AtomicBool,
    delete_error: Mutex<Option<String>>,
    next_key: Mutex<Option<String>>,
}

impl Default for MemoryMediaProvider {
    fn default() -> Self {
        Self {
            state: Mutex::default(),
            supports_reserved: AtomicBool::new(true),
            supports_reserved_files: AtomicBool::new(true),
            now_unix_ms: AtomicU64::new(0),
            reserved_put_gate: Mutex::default(),
            fail_deletes: AtomicBool::default(),
            delete_error: Mutex::default(),
            next_key: Mutex::default(),
        }
    }
}

impl MemoryMediaProvider {
    fn object_count(&self) -> usize {
        self.state.lock().unwrap().objects.len()
    }

    fn tamper_single_object(&self, bytes: Vec<u8>) {
        let mut state = self.state.lock().unwrap();
        let object = state.objects.values_mut().next().expect("one object");
        object.bytes = bytes;
        object.metadata.content_length = Some(object.bytes.len() as u64);
        object.metadata.sha256 = None;
    }

    fn remove_single_object(&self) {
        let mut state = self.state.lock().unwrap();
        let key = state.objects.keys().next().cloned().unwrap();
        state.objects.remove(&key);
    }

    fn set_delete_error(&self, error: String) {
        *self.delete_error.lock().unwrap() = Some(error);
    }

    fn clear_delete_error(&self) {
        *self.delete_error.lock().unwrap() = None;
    }

    fn set_next_key(&self, key: String) {
        *self.next_key.lock().unwrap() = Some(key);
    }

    fn set_now_unix_ms(&self, now: u64) {
        self.now_unix_ms.store(now, Ordering::SeqCst);
    }

    fn now_unix_ms(&self) -> u64 {
        let fixed = self.now_unix_ms.load(Ordering::SeqCst);
        if fixed != 0 {
            return fixed;
        }
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn set_reserved_put_gate(&self, entered: StdArc<Notify>, release: StdArc<Notify>) {
        *self.reserved_put_gate.lock().unwrap() = Some((entered, release));
    }
}

#[async_trait::async_trait]
impl MediaStorageProvider for MemoryMediaProvider {
    fn reserved_write_protocol_version(&self) -> Option<u16> {
        self.supports_reserved
            .load(Ordering::SeqCst)
            .then_some(MEDIA_RESERVED_WRITE_VERSION)
    }

    fn supports_reserved_file_writes(&self) -> bool {
        self.supports_reserved_files.load(Ordering::SeqCst)
    }

    async fn put(
        &self,
        request: MediaWriteRequest,
        bytes: Vec<u8>,
    ) -> HookResult<StoredMediaObject> {
        let key = self
            .next_key
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| format!("private/object/{}", request.record_id));
        let tenant_id = request.metadata.get("tenant_id").cloned();
        let metadata = MediaObjectMetadata {
            content_type: request.content_type,
            filename: request.preferred_filename,
            content_length: Some(bytes.len() as u64),
            sha256: Some(sha256_hex(&bytes)),
            tenant_id,
            attributes: request.metadata,
        };
        self.state.lock().unwrap().objects.insert(
            key.clone(),
            MemoryObject {
                bytes,
                metadata: metadata.clone(),
            },
        );
        Ok(StoredMediaObject {
            key: MediaObjectKey::new(key),
            metadata,
        })
    }

    async fn put_reserved(
        &self,
        request: MediaReservedWriteRequest,
        bytes: Vec<u8>,
    ) -> HookResult<StoredMediaObject> {
        let gate = self.reserved_put_gate.lock().unwrap().take();
        if let Some((entered, release)) = gate {
            entered.notify_one();
            release.notified().await;
        }
        let size_bytes = bytes.len() as u64;
        let sha256 = sha256_hex(&bytes);
        if request.version != MEDIA_RESERVED_WRITE_VERSION
            || uuid::Uuid::parse_str(&request.write_id).is_err()
            || request.request.record_id != request.write_id
            || self.now_unix_ms() >= request.expires_at_unix_ms
            || request.content_length != size_bytes
            || request.sha256 != sha256
        {
            return Err(HookError::Failed("invalid reserved write envelope".into()));
        }
        let mut state = self.state.lock().unwrap();
        if state.fenced_write_ids.contains(&request.write_id) {
            return Err(HookError::Failed("reserved write is fenced".into()));
        }
        if let Some(existing) = state.reserved_writes.get(&request.write_id) {
            if existing.request != request.request
                || existing.size_bytes != size_bytes
                || existing.sha256 != sha256
            {
                return Err(HookError::Failed(
                    "reserved write identity was reused".into(),
                ));
            }
            let object = state
                .objects
                .get(&existing.key)
                .cloned()
                .ok_or_else(|| HookError::Failed("reserved object disappeared".into()))?;
            return Ok(StoredMediaObject {
                key: MediaObjectKey::new(existing.key.clone()),
                metadata: object.metadata,
            });
        }
        let write_id = request.write_id.clone();
        let original_request = request.request.clone();
        let key = self
            .next_key
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| format!("private/object/{}", request.request.record_id));
        let tenant_id = request.request.metadata.get("tenant_id").cloned();
        let metadata = MediaObjectMetadata {
            content_type: request.request.content_type,
            filename: request.request.preferred_filename,
            content_length: Some(size_bytes),
            sha256: Some(sha256.clone()),
            tenant_id,
            attributes: request.request.metadata,
        };
        state.objects.insert(
            key.clone(),
            MemoryObject {
                bytes,
                metadata: metadata.clone(),
            },
        );
        state.reserved_writes.insert(
            write_id,
            MemoryReservedWrite {
                request: original_request,
                size_bytes,
                sha256,
                key: key.clone(),
            },
        );
        Ok(StoredMediaObject {
            key: MediaObjectKey::new(key),
            metadata,
        })
    }

    async fn put_reserved_file(
        &self,
        request: MediaReservedWriteRequest,
        path: PathBuf,
    ) -> HookResult<StoredMediaObject> {
        let mut source = tokio::fs::File::open(path)
            .await
            .map_err(|error| HookError::Failed(error.to_string()))?;
        let capacity = usize::try_from(request.content_length)
            .unwrap_or(0)
            .min(64 * 1024);
        let mut bytes = Vec::with_capacity(capacity);
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = source
                .read(&mut buffer)
                .await
                .map_err(|error| HookError::Failed(error.to_string()))?;
            if count == 0 {
                break;
            }
            if bytes.len().saturating_add(count) as u64 > request.content_length {
                return Err(HookError::Failed(
                    "reserved file exceeds declared length".into(),
                ));
            }
            bytes.extend_from_slice(&buffer[..count]);
        }
        self.put_reserved(request, bytes).await
    }

    async fn recover_reserved_write(
        &self,
        request: MediaReservedWriteRecoveryRequest,
    ) -> HookResult<()> {
        if request.version != MEDIA_RESERVED_WRITE_VERSION
            || uuid::Uuid::parse_str(&request.write_id).is_err()
            || request.request.record_id != request.write_id
            || self.now_unix_ms() < request.expires_at_unix_ms
        {
            return Err(HookError::Failed(
                "invalid reserved recovery envelope".into(),
            ));
        }
        if let Some(error) = self.delete_error.lock().unwrap().clone() {
            return Err(HookError::Failed(error));
        }
        if self.fail_deletes.load(Ordering::SeqCst) {
            return Err(HookError::Failed("injected delete failure".to_string()));
        }
        let mut state = self.state.lock().unwrap();
        if state
            .reserved_writes
            .get(&request.write_id)
            .is_some_and(|write| {
                write.request != request.request
                    || write.size_bytes != request.content_length
                    || write.sha256 != request.sha256
            })
        {
            return Err(HookError::Failed(
                "reserved recovery did not match its write identity".into(),
            ));
        }
        state.fenced_write_ids.insert(request.write_id.clone());
        let key = state
            .reserved_writes
            .remove(&request.write_id)
            .map(|write| write.key)
            .or_else(|| request.storage_key.map(|key| key.key));
        match key.and_then(|key| state.objects.remove(&key)) {
            Some(_) => Ok(()),
            None => Err(HookError::NotFound(
                "reserved write is absent and fenced".to_string(),
            )),
        }
    }

    async fn get(&self, request: MediaReadRequest) -> HookResult<izwi_hooks::StoredMediaBytes> {
        let object = self
            .state
            .lock()
            .unwrap()
            .objects
            .get(&request.key.key)
            .cloned()
            .ok_or_else(|| HookError::NotFound("object".to_string()))?;
        Ok(izwi_hooks::StoredMediaBytes {
            bytes: object.bytes,
            metadata: object.metadata,
        })
    }

    async fn get_stream(&self, request: MediaReadRequest) -> HookResult<StoredMediaStream> {
        let object = self
            .state
            .lock()
            .unwrap()
            .objects
            .get(&request.key.key)
            .cloned()
            .ok_or_else(|| HookError::NotFound("object".to_string()))?;
        Ok(StoredMediaStream {
            reader: Box::pin(Cursor::new(object.bytes)),
            metadata: object.metadata,
        })
    }

    async fn delete(&self, request: MediaDeleteRequest) -> HookResult<()> {
        if let Some(error) = self.delete_error.lock().unwrap().clone() {
            return Err(HookError::Failed(error));
        }
        if self.fail_deletes.load(Ordering::SeqCst) {
            return Err(HookError::Failed("injected delete failure".to_string()));
        }
        if self
            .state
            .lock()
            .unwrap()
            .objects
            .remove(&request.key.key)
            .is_some()
        {
            Ok(())
        } else {
            Err(HookError::NotFound("object".to_string()))
        }
    }
}

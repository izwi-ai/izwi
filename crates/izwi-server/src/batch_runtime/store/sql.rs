//! Column lists, row mapping and SQL value helpers shared by the store.

use super::*;

pub(crate) const DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT: usize = 64;

pub(super) const MAX_RUNTIME_MAINTENANCE_BATCH_LIMIT: usize = 512;

pub(super) fn bounded_maintenance_batch_limit(limit: usize) -> usize {
    limit.clamp(1, MAX_RUNTIME_MAINTENANCE_BATCH_LIMIT)
}

pub(super) fn validate_bounded_field(name: &str, value: &str, max_bytes: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty(),
        "Durable idempotency {name} cannot be empty"
    );
    anyhow::ensure!(
        value.len() <= max_bytes,
        "Durable idempotency {name} exceeds {max_bytes} bytes"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "Durable idempotency {name} contains control characters"
    );
    Ok(())
}

pub(super) fn validate_optional_field_bytes(
    name: &str,
    value: Option<&str>,
    max_bytes: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.is_none_or(|value| value.len() <= max_bytes),
        StoreError::invalid(format!("{name} exceeds {max_bytes} bytes"))
    );
    Ok(())
}

pub(super) fn nonnegative_timestamp(now: i64) -> anyhow::Result<u64> {
    u64::try_from(now).context("Durable idempotency clock preceded the Unix epoch")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn current_timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// `SELECT <every column> FROM <table>` for each mapped row type, to
/// `concat!` into complete statements.
macro_rules! select_media_assets {
    () => {
        concat!(
            "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, ",
            "content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, ",
            "channel_count, peak_amplitude, rms_amplitude, source_asset_id, ",
            "canonical_profile_version, scan_status, retention_policy, deleted_at, ",
            "metadata_json",
            " FROM media_assets"
        )
    };
}

macro_rules! select_provider_write_reservations {
    () => {
        concat!(
            "SELECT write_id, reservation_token, created_at, expires_at, tenant_scope, ",
            "storage_namespace, content_type, filename, expected_size_bytes, expected_sha256, ",
            "provider_request_json, storage_key, cleanup_claim_token, cleanup_attempt_count",
            " FROM provider_write_reservations"
        )
    };
}
pub(super) use select_provider_write_reservations;

macro_rules! select_text_assets {
    () => {
        concat!(
            "SELECT id, created_at, updated_at, raw_text, normalized_text, language_hint, ",
            "character_count, sha256, safety_status, retention_policy, structure_json",
            " FROM text_assets"
        )
    };
}

macro_rules! select_runtime_jobs {
    () => {
        concat!(
            "SELECT id, created_at, updated_at, queued_at, started_at, finished_at, job_kind, status, ",
            "priority, model_id, capability, route_record_kind, route_record_id, ",
            "input_media_asset_id, input_text_asset_id, request_json, model_snapshot_json, ",
            "progress_json, error_code, error_message, attempt_count, max_attempts, ",
            "retry_policy_json, idempotency_key, correlation_id, cancellation_reason, ",
            "cancellation_state",
            " FROM runtime_jobs"
        )
    };
}
pub(super) use select_runtime_jobs;

macro_rules! select_job_stages {
    () => {
        concat!(
            "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, ",
            "resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, ",
            "available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, ",
            "output_artifact_ids_json, progress_json, started_at, finished_at, error_code, ",
            "error_message, cancellation_state",
            " FROM job_stages"
        )
    };
}

macro_rules! select_runtime_artifacts {
    () => {
        concat!(
            "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, ",
            "publication_key, created_at, artifact_kind, artifact_role, media_asset_id, ",
            "text_asset_id, storage_key, content_type, filename, size_bytes, sha256, ",
            "metadata_json, retention_policy",
            " FROM runtime_artifacts"
        )
    };
}
pub(super) use select_runtime_artifacts;

macro_rules! select_worker_heartbeats {
    () => {
        concat!(
            "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, instance_id, ",
            "registration_version, registration_json, heartbeat_version, available_slots, ",
            "heartbeat_details_json, current_job_id, current_stage_id, diagnostic_json",
            " FROM runtime_worker_heartbeats"
        )
    };
}

pub(super) const MEDIA_ASSET_BY_ID_SQL: &str = concat!(select_media_assets!(), " WHERE id = ?1");

pub(super) const MEDIA_ASSET_BY_STORAGE_KEY_SQL: &str = concat!(
    select_media_assets!(),
    " WHERE storage_key = ?1 AND deleted_at IS NULL"
);

pub(super) const MEDIA_ASSET_BY_SOURCE_PROFILE_SQL: &str = concat!(
    select_media_assets!(),
    " WHERE source_asset_id = ?1 AND canonical_profile_version = ?2 AND deleted_at IS NULL"
);

pub(super) const TEXT_ASSET_BY_ID_SQL: &str = concat!(select_text_assets!(), " WHERE id = ?1");

pub(super) const RUNTIME_JOB_BY_ID_SQL: &str = concat!(select_runtime_jobs!(), " WHERE id = ?1");

pub(super) const JOB_STAGE_BY_ID_SQL: &str = concat!(select_job_stages!(), " WHERE id = ?1");

pub(super) const JOB_STAGE_LIST_FOR_JOB_SQL: &str = concat!(
    select_job_stages!(),
    " WHERE job_id = ?1 ORDER BY sequence ASC, created_at ASC, id ASC"
);

pub(super) const RUNTIME_ARTIFACT_BY_ID_SQL: &str =
    concat!(select_runtime_artifacts!(), " WHERE id = ?1");

pub(super) const RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL: &str = concat!(
    select_runtime_artifacts!(),
    " WHERE job_id = ?1 ORDER BY created_at ASC, id ASC"
);

pub(super) const WORKER_HEARTBEAT_BY_ID_SQL: &str =
    concat!(select_worker_heartbeats!(), " WHERE worker_id = ?1");

pub(super) const RUNTIME_WORKER_HEARTBEATS_SQL: &str =
    concat!(select_worker_heartbeats!(), " ORDER BY worker_id");

/// One column of a raw query row, by name. Rows map by name rather than by
/// position, so a statement may list its columns in any order.
pub(super) fn col<T: sea_orm::TryGetable>(row: &QueryResult, name: &str) -> anyhow::Result<T> {
    row.try_get::<T>("", name)
        .with_context(|| format!("Failed to read column {name}"))
}

/// Serialize one capacity check across processes: SQLite through its writer
/// lock, PostgreSQL through the row lock the no-op update holds until the
/// transaction ends.
pub(super) async fn lock_named<C: ConnectionTrait>(db: &C, name: &str) -> anyhow::Result<()> {
    db.execute_raw(raw::statement(
        db,
        "INSERT INTO runtime_admission_locks (id, lock_value) VALUES (?1, 1) ON CONFLICT (id) DO NOTHING",
        vec![name.into()],
    )?)
    .await
    .with_context(|| format!("Failed to initialize the {name} admission lock"))?;
    db.execute_raw(raw::statement(
        db,
        "UPDATE runtime_admission_locks SET lock_value = lock_value WHERE id = ?1",
        vec![name.into()],
    )?)
    .await
    .with_context(|| format!("Failed to take the {name} admission lock"))?;
    Ok(())
}

pub(super) const TTS_ADMISSION_LOCK: &str = "tts";

pub(super) fn map_media_asset(row: &QueryResult) -> anyhow::Result<MediaAsset> {
    Ok(MediaAsset {
        id: col(row, "id")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        updated_at: i64_to_u64(col(row, "updated_at")?)?,
        asset_kind: col(row, "asset_kind")?,
        storage_namespace: col(row, "storage_namespace")?,
        storage_key: col(row, "storage_key")?,
        content_type: col(row, "content_type")?,
        filename: col(row, "filename")?,
        size_bytes: i64_to_u64(col(row, "size_bytes")?)?,
        sha256: col(row, "sha256")?,
        duration_secs: col(row, "duration_secs")?,
        sample_rate_hz: opt_i64_to_u32(col(row, "sample_rate_hz")?)?,
        channel_count: opt_i64_to_u16(col(row, "channel_count")?)?,
        peak_amplitude: col::<Option<f64>>(row, "peak_amplitude")?
            .map(|value| value as f32),
        rms_amplitude: col::<Option<f64>>(row, "rms_amplitude")?
            .map(|value| value as f32),
        source_asset_id: col(row, "source_asset_id")?,
        canonical_profile_version: col(row, "canonical_profile_version")?,
        scan_status: col(row, "scan_status")?,
        retention_policy: col(row, "retention_policy")?,
        deleted_at: opt_i64_to_u64(col(row, "deleted_at")?)?,
        metadata_json: parse_json_value(col::<String>(row, "metadata_json")?, json!({})),
    })
}

pub(super) fn map_artifact_cleanup_intent(row: &QueryResult) -> anyhow::Result<ArtifactCleanupIntent> {
    let storage_key: String = col(row, "storage_key")?;
    let tenant_scope: String = col(row, "tenant_scope")?;
    validate_artifact_cleanup_storage_key(&storage_key)?;
    validate_artifact_cleanup_tenant(&tenant_scope)?;
    let last_error: Option<String> = col(row, "last_error")?;
    anyhow::ensure!(
        last_error
            .as_ref()
            .is_none_or(|error| error.len() <= MAX_ARTIFACT_CLEANUP_ERROR_BYTES),
        "Stored artifact cleanup error exceeds its bound"
    );
    Ok(ArtifactCleanupIntent {
        id: col(row, "id")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        updated_at: i64_to_u64(col(row, "updated_at")?)?,
        available_at: i64_to_u64(col(row, "available_at")?)?,
        storage_key,
        tenant_scope,
        reason: ArtifactCleanupReason::from_db_value(&col::<String>(row, "reason")?)?,
        attempt_count: i64_to_u32(col(row, "attempt_count")?)?,
        last_error,
    })
}

pub(super) fn map_provider_write(row: &QueryResult) -> anyhow::Result<ProviderWriteReservation> {
    let write_id: String = col(row, "write_id")?;
    let tenant_scope: String = col(row, "tenant_scope")?;
    let storage_namespace: String = col(row, "storage_namespace")?;
    let content_type: String = col(row, "content_type")?;
    let filename: Option<String> = col(row, "filename")?;
    let provider_request = match col::<Option<String>>(row, "provider_request_json")? {
        Some(raw) => deserialize_provider_write_request(&raw)?,
        None => legacy_provider_write_request(
            &write_id,
            &tenant_scope,
            &storage_namespace,
            &content_type,
            filename.as_deref(),
        )?,
    };
    let reservation = ProviderWriteReservation {
        write_id,
        reservation_token: col(row, "reservation_token")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        expires_at: i64_to_u64(col(row, "expires_at")?)?,
        tenant_scope,
        storage_namespace,
        content_type,
        filename,
        expected_size_bytes: i64_to_u64(col(row, "expected_size_bytes")?)?,
        expected_sha256: col(row, "expected_sha256")?,
        provider_request,
        storage_key: col(row, "storage_key")?,
        cleanup_claim_token: col(row, "cleanup_claim_token")?,
        cleanup_attempt_count: i64_to_u32(col(row, "cleanup_attempt_count")?)?,
    };
    anyhow::ensure!(
        uuid::Uuid::parse_str(&reservation.write_id).is_ok()
            && uuid::Uuid::parse_str(&reservation.reservation_token).is_ok()
            && reservation
                .cleanup_claim_token
                .as_deref()
                .is_none_or(|token| { uuid::Uuid::parse_str(token).is_ok() }),
        "Stored provider write identity is invalid"
    );
    validate_provider_write_input(&NewProviderWriteReservation {
        write_id: reservation.write_id.clone(),
        tenant_scope: reservation.tenant_scope.clone(),
        storage_namespace: reservation.storage_namespace.clone(),
        content_type: reservation.content_type.clone(),
        filename: reservation.filename.clone(),
        expected_size_bytes: reservation.expected_size_bytes,
        expected_sha256: reservation.expected_sha256.clone(),
        lifetime_ms: reservation
            .expires_at
            .saturating_sub(reservation.created_at),
        provider_request: reservation.provider_request.clone(),
    })?;
    if let Some(key) = reservation.storage_key.as_deref() {
        validate_artifact_cleanup_storage_key(key)?;
    }
    Ok(reservation)
}

pub(super) fn map_text_asset(row: &QueryResult) -> anyhow::Result<TextAsset> {
    Ok(TextAsset {
        id: col(row, "id")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        updated_at: i64_to_u64(col(row, "updated_at")?)?,
        raw_text: col(row, "raw_text")?,
        normalized_text: col(row, "normalized_text")?,
        language_hint: col(row, "language_hint")?,
        character_count: i64_to_u64(col(row, "character_count")?)?,
        sha256: col(row, "sha256")?,
        safety_status: col(row, "safety_status")?,
        retention_policy: col(row, "retention_policy")?,
        structure_json: parse_json_value(col::<String>(row, "structure_json")?, json!({})),
    })
}

pub(super) fn map_runtime_job(row: &QueryResult) -> anyhow::Result<RuntimeJob> {
    let kind_raw: String = col(row, "job_kind")?;
    let status_raw: String = col(row, "status")?;

    Ok(RuntimeJob {
        id: col(row, "id")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        updated_at: i64_to_u64(col(row, "updated_at")?)?,
        queued_at: opt_i64_to_u64(col(row, "queued_at")?)?,
        started_at: opt_i64_to_u64(col(row, "started_at")?)?,
        finished_at: opt_i64_to_u64(col(row, "finished_at")?)?,
        job_kind: RuntimeJobKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {kind_raw}"))?,
        status: RuntimeJobStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job status: {status_raw}"))?,
        priority: i64_to_i32(col(row, "priority")?)?,
        model_id: col(row, "model_id")?,
        capability: col(row, "capability")?,
        route_record_kind: col(row, "route_record_kind")?,
        route_record_id: col(row, "route_record_id")?,
        input_media_asset_id: col(row, "input_media_asset_id")?,
        input_text_asset_id: col(row, "input_text_asset_id")?,
        request_json: parse_json_value(col::<String>(row, "request_json")?, json!({})),
        model_snapshot_json: parse_json_value(col::<String>(row, "model_snapshot_json")?, json!({})),
        progress_json: col::<Option<String>>(row, "progress_json")?
            .map(|raw| parse_json_value(raw, json!({}))),
        error_code: col(row, "error_code")?,
        error_message: col(row, "error_message")?,
        attempt_count: i64_to_u32(col(row, "attempt_count")?)?,
        max_attempts: i64_to_u32(col(row, "max_attempts")?)?,
        retry_policy_json: parse_json_value(col::<String>(row, "retry_policy_json")?, json!({})),
        idempotency_key: col(row, "idempotency_key")?,
        correlation_id: col(row, "correlation_id")?,
        cancellation_reason: col(row, "cancellation_reason")?,
        cancellation_state: parse_cancellation_state(col(row, "cancellation_state")?)?,
    })
}

pub(super) fn map_job_stage(row: &QueryResult) -> anyhow::Result<JobStage> {
    let queue_class_raw: String = col(row, "queue_class")?;
    let status_raw: String = col(row, "status")?;

    Ok(JobStage {
        id: col(row, "id")?,
        job_id: col(row, "job_id")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        updated_at: i64_to_u64(col(row, "updated_at")?)?,
        sequence: i64_to_u32(col(row, "sequence")?)?,
        stage_kind: col(row, "stage_kind")?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(col(row, "resource_hints_json")?),
        status: RuntimeStageStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?,
        capability: col(row, "capability")?,
        model_id: col(row, "model_id")?,
        worker_id: col(row, "worker_id")?,
        lease_expires_at: opt_i64_to_u64(col(row, "lease_expires_at")?)?,
        available_at: opt_i64_to_u64(col(row, "available_at")?)?,
        attempt_token: col(row, "attempt_token")?,
        attempt_count: i64_to_u32(col(row, "attempt_count")?)?,
        max_attempts: i64_to_u32(col(row, "max_attempts")?)?,
        input_artifact_ids: parse_string_array(col::<String>(row, "input_artifact_ids_json")?),
        output_artifact_ids: parse_string_array(col::<String>(row, "output_artifact_ids_json")?),
        progress_json: col::<Option<String>>(row, "progress_json")?
            .map(|raw| parse_json_value(raw, json!({}))),
        started_at: opt_i64_to_u64(col(row, "started_at")?)?,
        finished_at: opt_i64_to_u64(col(row, "finished_at")?)?,
        error_code: col(row, "error_code")?,
        error_message: col(row, "error_message")?,
        cancellation_state: parse_cancellation_state(col(row, "cancellation_state")?)?,
    })
}

pub(super) fn parse_cancellation_state(
    value: Option<String>,
) -> anyhow::Result<Option<RuntimeCancellationState>> {
    value
        .map(|value| {
            RuntimeCancellationState::from_db_value(&value)
                .ok_or_else(|| anyhow!("Unknown runtime cancellation state: {value}"))
        })
        .transpose()
}

pub(super) fn map_stage_claim_candidate(row: &QueryResult) -> anyhow::Result<StageClaimCandidate> {
    let job_kind_raw: String = col(row, "job_kind")?;
    let queue_class_raw: String = col(row, "queue_class")?;

    Ok(StageClaimCandidate {
        stage_id: col(row, "id")?,
        stage_kind: col(row, "stage_kind")?,
        job_kind: RuntimeJobKind::from_db_value(job_kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {job_kind_raw}"))?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(col(row, "resource_hints_json")?),
        capability: col(row, "capability")?,
        model_id: col(row, "model_id")?,
    })
}

pub(super) fn map_runtime_artifact(row: &QueryResult) -> anyhow::Result<RuntimeArtifact> {
    let kind_raw: String = col(row, "artifact_kind")?;
    let role_raw: String = col(row, "artifact_role")?;

    Ok(RuntimeArtifact {
        id: col(row, "id")?,
        job_id: col(row, "job_id")?,
        stage_id: col(row, "stage_id")?,
        producer_attempt_count: opt_i64_to_u32(col(row, "producer_attempt_count")?)?,
        producer_attempt_token: col(row, "producer_attempt_token")?,
        publication_key: col(row, "publication_key")?,
        created_at: i64_to_u64(col(row, "created_at")?)?,
        artifact_kind: RuntimeArtifactKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact kind: {kind_raw}"))?,
        artifact_role: RuntimeArtifactRole::from_db_value(role_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact role: {role_raw}"))?,
        media_asset_id: col(row, "media_asset_id")?,
        text_asset_id: col(row, "text_asset_id")?,
        storage_key: col(row, "storage_key")?,
        content_type: col(row, "content_type")?,
        filename: col(row, "filename")?,
        size_bytes: opt_i64_to_u64(col(row, "size_bytes")?)?,
        sha256: col(row, "sha256")?,
        metadata_json: parse_json_value(col::<String>(row, "metadata_json")?, json!({})),
        retention_policy: col(row, "retention_policy")?,
    })
}

pub(super) fn map_worker_heartbeat(row: &QueryResult) -> anyhow::Result<RuntimeWorkerHeartbeat> {
    let worker_id: String = col(row, "worker_id")?;
    let queue_names = parse_string_array(col::<String>(row, "queue_names_json")?);
    let stored_instance_id: String = col(row, "instance_id")?;
    let instance_id = if stored_instance_id.is_empty() {
        worker_id.clone()
    } else {
        stored_instance_id
    };
    let registration_version = i64_to_u32(col(row, "registration_version")?)? as u16;
    let registration =
        serde_json::from_str::<RuntimeWorkerRegistration>(&col::<String>(row, "registration_json")?)
            .unwrap_or_else(|_| RuntimeWorkerRegistration {
                version: registration_version,
                worker_id: worker_id.clone(),
                instance_id: instance_id.clone(),
                queue_classes: queue_names
                    .iter()
                    .filter_map(|queue| QueueClass::from_db_value(queue))
                    .collect(),
                capabilities: Vec::new(),
                model_ids: Vec::new(),
                stage_kinds: Vec::new(),
                resources: WorkerResourceCapacity::default(),
                software_version: "legacy".to_string(),
            });
    let heartbeat_version = i64_to_u32(col(row, "heartbeat_version")?)? as u16;
    let available_slots = i64_to_u32(col(row, "available_slots")?)?;
    let details =
        serde_json::from_str::<RuntimeWorkerHeartbeatDetails>(&col::<String>(row, "heartbeat_details_json")?)
            .unwrap_or_else(|_| RuntimeWorkerHeartbeatDetails {
                version: heartbeat_version,
                available_slots,
                active_lease_ids: Vec::new(),
                last_error: None,
                health_json: json!({}),
            });

    Ok(RuntimeWorkerHeartbeat {
        worker_id,
        started_at: i64_to_u64(col(row, "started_at")?)?,
        last_heartbeat_at: i64_to_u64(col(row, "last_heartbeat_at")?)?,
        status: col(row, "status")?,
        queue_names,
        instance_id,
        registration,
        details,
        current_job_id: col(row, "current_job_id")?,
        current_stage_id: col(row, "current_stage_id")?,
        diagnostic_json: parse_json_value(col::<String>(row, "diagnostic_json")?, json!({})),
    })
}

pub(super) fn json_to_db_string(value: &serde_json::Value, fallback: &str) -> anyhow::Result<String> {
    serde_json::to_string(value)
        .or_else(|_| Ok::<String, serde_json::Error>(fallback.to_string()))
        .context("Failed to serialize runtime JSON payload")
}

pub(super) fn parse_json_value(raw: String, fallback: serde_json::Value) -> serde_json::Value {
    serde_json::from_str(raw.as_str()).unwrap_or(fallback)
}

pub(super) fn parse_string_array(raw: String) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw.as_str()).unwrap_or_default()
}

pub(super) fn parse_resource_hints(raw: String) -> StageResourceHints {
    serde_json::from_str::<StageResourceHints>(&raw).unwrap_or_default()
}

pub(super) fn opt_string(value: Option<String>) -> Value {
    Value::String(value)
}

pub(super) fn opt_u64(value: Option<u64>) -> Value {
    Value::BigInt(value.and_then(|value| i64::try_from(value).ok()))
}

pub(super) fn opt_i64(value: Option<i64>) -> Value {
    Value::BigInt(value)
}

pub(super) fn opt_u32(value: Option<u32>) -> Value {
    Value::BigInt(value.map(i64::from))
}

pub(super) fn opt_u16(value: Option<u16>) -> Value {
    Value::BigInt(value.map(i64::from))
}

pub(super) fn opt_f64(value: Option<f64>) -> Value {
    Value::Double(value)
}

pub(super) fn opt_f32(value: Option<f32>) -> Value {
    Value::Double(value.map(f64::from))
}

pub(super) fn u64_to_i64_value(value: u64) -> anyhow::Result<Value> {
    Ok(Value::BigInt(Some(i64::try_from(value)?)))
}

pub(super) fn u32_to_i64_value(value: u32) -> i64 {
    i64::from(value)
}

pub(super) fn i64_to_u64(value: i64) -> anyhow::Result<u64> {
    u64::try_from(value).map_err(Into::into)
}

pub(super) fn opt_i64_to_u64(value: Option<i64>) -> anyhow::Result<Option<u64>> {
    value.map(i64_to_u64).transpose()
}

pub(super) fn i64_to_u32(value: i64) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(Into::into)
}

pub(super) fn opt_i64_to_u32(value: Option<i64>) -> anyhow::Result<Option<u32>> {
    value.map(i64_to_u32).transpose()
}

pub(super) fn opt_i64_to_u16(value: Option<i64>) -> anyhow::Result<Option<u16>> {
    value
        .map(|value| u16::try_from(value).map_err(Into::into))
        .transpose()
}

pub(super) fn i64_to_i32(value: i64) -> anyhow::Result<i32> {
    i32::try_from(value).map_err(Into::into)
}

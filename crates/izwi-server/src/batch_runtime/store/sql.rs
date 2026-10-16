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

pub(super) const MEDIA_ASSET_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, source_asset_id, canonical_profile_version, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE id = ?1";

pub(super) const PROVIDER_WRITE_COLUMNS_SQL: &str = "SELECT write_id, reservation_token, created_at, expires_at, tenant_scope, storage_namespace, content_type, filename, expected_size_bytes, expected_sha256, provider_request_json, storage_key, cleanup_claim_token, cleanup_attempt_count FROM provider_write_reservations WHERE write_id = ?1";

pub(super) const MEDIA_ASSET_BY_STORAGE_KEY_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, source_asset_id, canonical_profile_version, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE storage_key = ?1 AND deleted_at IS NULL";

pub(super) const MEDIA_ASSET_BY_SOURCE_PROFILE_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, source_asset_id, canonical_profile_version, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE source_asset_id = ?1 AND canonical_profile_version = ?2 AND deleted_at IS NULL";

pub(super) const TEXT_ASSET_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, raw_text, normalized_text, language_hint, character_count, sha256, safety_status, retention_policy, structure_json FROM text_assets WHERE id = ?1";

pub(super) const RUNTIME_JOB_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, queued_at, started_at, finished_at, job_kind, status, priority, model_id, capability, route_record_kind, route_record_id, input_media_asset_id, input_text_asset_id, request_json, model_snapshot_json, progress_json, error_code, error_message, attempt_count, max_attempts, retry_policy_json, idempotency_key, correlation_id, cancellation_reason, cancellation_state FROM runtime_jobs WHERE id = ?1";

pub(super) const JOB_STAGE_COLUMNS_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message, cancellation_state FROM job_stages WHERE id = ?1";

pub(super) const JOB_STAGE_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message, cancellation_state FROM job_stages WHERE job_id = ?1 ORDER BY sequence ASC, created_at ASC, id ASC";

pub(super) const RUNTIME_ARTIFACT_COLUMNS_SQL: &str =
    "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, publication_key, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE id = ?1";

pub(super) const RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, publication_key, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE job_id = ?1 ORDER BY created_at ASC, id ASC";

pub(super) const WORKER_HEARTBEAT_COLUMNS_SQL: &str =
    "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, instance_id, registration_version, registration_json, heartbeat_version, available_slots, heartbeat_details_json, current_job_id, current_stage_id, diagnostic_json FROM runtime_worker_heartbeats WHERE worker_id = ?1";

pub(super) const RUNTIME_WORKER_HEARTBEATS_SQL: &str =
    "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, instance_id, registration_version, registration_json, heartbeat_version, available_slots, heartbeat_details_json, current_job_id, current_stage_id, diagnostic_json FROM runtime_worker_heartbeats ORDER BY worker_id";

pub(super) fn map_media_asset(row: &QueryResult) -> anyhow::Result<MediaAsset> {
    Ok(MediaAsset {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        asset_kind: row.try_get_by_index(3)?,
        storage_namespace: row.try_get_by_index(4)?,
        storage_key: row.try_get_by_index(5)?,
        content_type: row.try_get_by_index(6)?,
        filename: row.try_get_by_index(7)?,
        size_bytes: i64_to_u64(row.try_get_by_index(8)?)?,
        sha256: row.try_get_by_index(9)?,
        duration_secs: row.try_get_by_index(10)?,
        sample_rate_hz: opt_i64_to_u32(row.try_get_by_index(11)?)?,
        channel_count: opt_i64_to_u16(row.try_get_by_index(12)?)?,
        peak_amplitude: row
            .try_get_by_index::<Option<f64>>(13)?
            .map(|value| value as f32),
        rms_amplitude: row
            .try_get_by_index::<Option<f64>>(14)?
            .map(|value| value as f32),
        source_asset_id: row.try_get_by_index(15)?,
        canonical_profile_version: row.try_get_by_index(16)?,
        scan_status: row.try_get_by_index(17)?,
        retention_policy: row.try_get_by_index(18)?,
        deleted_at: opt_i64_to_u64(row.try_get_by_index(19)?)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(20)?, json!({})),
    })
}

pub(super) fn map_artifact_cleanup_intent(row: &QueryResult) -> anyhow::Result<ArtifactCleanupIntent> {
    let storage_key: String = row.try_get_by_index(4)?;
    let tenant_scope: String = row.try_get_by_index(5)?;
    validate_artifact_cleanup_storage_key(&storage_key)?;
    validate_artifact_cleanup_tenant(&tenant_scope)?;
    let last_error: Option<String> = row.try_get_by_index(8)?;
    anyhow::ensure!(
        last_error
            .as_ref()
            .is_none_or(|error| error.len() <= MAX_ARTIFACT_CLEANUP_ERROR_BYTES),
        "Stored artifact cleanup error exceeds its bound"
    );
    Ok(ArtifactCleanupIntent {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        available_at: i64_to_u64(row.try_get_by_index(3)?)?,
        storage_key,
        tenant_scope,
        reason: ArtifactCleanupReason::from_db_value(&row.try_get_by_index::<String>(6)?)?,
        attempt_count: i64_to_u32(row.try_get_by_index(7)?)?,
        last_error,
    })
}

pub(super) fn map_provider_write(row: &QueryResult) -> anyhow::Result<ProviderWriteReservation> {
    let write_id: String = row.try_get_by_index(0)?;
    let tenant_scope: String = row.try_get_by_index(4)?;
    let storage_namespace: String = row.try_get_by_index(5)?;
    let content_type: String = row.try_get_by_index(6)?;
    let filename: Option<String> = row.try_get_by_index(7)?;
    let provider_request = match row.try_get_by_index::<Option<String>>(10)? {
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
        reservation_token: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        expires_at: i64_to_u64(row.try_get_by_index(3)?)?,
        tenant_scope,
        storage_namespace,
        content_type,
        filename,
        expected_size_bytes: i64_to_u64(row.try_get_by_index(8)?)?,
        expected_sha256: row.try_get_by_index(9)?,
        provider_request,
        storage_key: row.try_get_by_index(11)?,
        cleanup_claim_token: row.try_get_by_index(12)?,
        cleanup_attempt_count: i64_to_u32(row.try_get_by_index(13)?)?,
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
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        raw_text: row.try_get_by_index(3)?,
        normalized_text: row.try_get_by_index(4)?,
        language_hint: row.try_get_by_index(5)?,
        character_count: i64_to_u64(row.try_get_by_index(6)?)?,
        sha256: row.try_get_by_index(7)?,
        safety_status: row.try_get_by_index(8)?,
        retention_policy: row.try_get_by_index(9)?,
        structure_json: parse_json_value(row.try_get_by_index::<String>(10)?, json!({})),
    })
}

pub(super) fn map_runtime_job(row: &QueryResult) -> anyhow::Result<RuntimeJob> {
    let kind_raw: String = row.try_get_by_index(6)?;
    let status_raw: String = row.try_get_by_index(7)?;

    Ok(RuntimeJob {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        queued_at: opt_i64_to_u64(row.try_get_by_index(3)?)?,
        started_at: opt_i64_to_u64(row.try_get_by_index(4)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(5)?)?,
        job_kind: RuntimeJobKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {kind_raw}"))?,
        status: RuntimeJobStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job status: {status_raw}"))?,
        priority: i64_to_i32(row.try_get_by_index(8)?)?,
        model_id: row.try_get_by_index(9)?,
        capability: row.try_get_by_index(10)?,
        route_record_kind: row.try_get_by_index(11)?,
        route_record_id: row.try_get_by_index(12)?,
        input_media_asset_id: row.try_get_by_index(13)?,
        input_text_asset_id: row.try_get_by_index(14)?,
        request_json: parse_json_value(row.try_get_by_index::<String>(15)?, json!({})),
        model_snapshot_json: parse_json_value(row.try_get_by_index::<String>(16)?, json!({})),
        progress_json: row
            .try_get_by_index::<Option<String>>(17)?
            .map(|raw| parse_json_value(raw, json!({}))),
        error_code: row.try_get_by_index(18)?,
        error_message: row.try_get_by_index(19)?,
        attempt_count: i64_to_u32(row.try_get_by_index(20)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(21)?)?,
        retry_policy_json: parse_json_value(row.try_get_by_index::<String>(22)?, json!({})),
        idempotency_key: row.try_get_by_index(23)?,
        correlation_id: row.try_get_by_index(24)?,
        cancellation_reason: row.try_get_by_index(25)?,
        cancellation_state: parse_cancellation_state(row.try_get_by_index(26)?)?,
    })
}

pub(super) fn map_job_stage(row: &QueryResult) -> anyhow::Result<JobStage> {
    let queue_class_raw: String = row.try_get_by_index(6)?;
    let status_raw: String = row.try_get_by_index(8)?;

    Ok(JobStage {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(3)?)?,
        sequence: i64_to_u32(row.try_get_by_index(4)?)?,
        stage_kind: row.try_get_by_index(5)?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(row.try_get_by_index(7)?),
        status: RuntimeStageStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?,
        capability: row.try_get_by_index(9)?,
        model_id: row.try_get_by_index(10)?,
        worker_id: row.try_get_by_index(11)?,
        lease_expires_at: opt_i64_to_u64(row.try_get_by_index(12)?)?,
        available_at: opt_i64_to_u64(row.try_get_by_index(13)?)?,
        attempt_token: row.try_get_by_index(14)?,
        attempt_count: i64_to_u32(row.try_get_by_index(15)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(16)?)?,
        input_artifact_ids: parse_string_array(row.try_get_by_index::<String>(17)?),
        output_artifact_ids: parse_string_array(row.try_get_by_index::<String>(18)?),
        progress_json: row
            .try_get_by_index::<Option<String>>(19)?
            .map(|raw| parse_json_value(raw, json!({}))),
        started_at: opt_i64_to_u64(row.try_get_by_index(20)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(21)?)?,
        error_code: row.try_get_by_index(22)?,
        error_message: row.try_get_by_index(23)?,
        cancellation_state: parse_cancellation_state(row.try_get_by_index(24)?)?,
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
    let job_kind_raw: String = row.try_get_by_index(2)?;
    let queue_class_raw: String = row.try_get_by_index(5)?;

    Ok(StageClaimCandidate {
        stage_id: row.try_get_by_index(0)?,
        stage_kind: row.try_get_by_index(1)?,
        job_kind: RuntimeJobKind::from_db_value(job_kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {job_kind_raw}"))?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(row.try_get_by_index(6)?),
        capability: row.try_get_by_index(3)?,
        model_id: row.try_get_by_index(4)?,
    })
}

pub(super) fn map_runtime_artifact(row: &QueryResult) -> anyhow::Result<RuntimeArtifact> {
    let kind_raw: String = row.try_get_by_index(7)?;
    let role_raw: String = row.try_get_by_index(8)?;

    Ok(RuntimeArtifact {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        stage_id: row.try_get_by_index(2)?,
        producer_attempt_count: opt_i64_to_u32(row.try_get_by_index(3)?)?,
        producer_attempt_token: row.try_get_by_index(4)?,
        publication_key: row.try_get_by_index(5)?,
        created_at: i64_to_u64(row.try_get_by_index(6)?)?,
        artifact_kind: RuntimeArtifactKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact kind: {kind_raw}"))?,
        artifact_role: RuntimeArtifactRole::from_db_value(role_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact role: {role_raw}"))?,
        media_asset_id: row.try_get_by_index(9)?,
        text_asset_id: row.try_get_by_index(10)?,
        storage_key: row.try_get_by_index(11)?,
        content_type: row.try_get_by_index(12)?,
        filename: row.try_get_by_index(13)?,
        size_bytes: opt_i64_to_u64(row.try_get_by_index(14)?)?,
        sha256: row.try_get_by_index(15)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(16)?, json!({})),
        retention_policy: row.try_get_by_index(17)?,
    })
}

pub(super) fn map_worker_heartbeat(row: &QueryResult) -> anyhow::Result<RuntimeWorkerHeartbeat> {
    let worker_id: String = row.try_get_by_index(0)?;
    let queue_names = parse_string_array(row.try_get_by_index::<String>(4)?);
    let stored_instance_id: String = row.try_get_by_index(5)?;
    let instance_id = if stored_instance_id.is_empty() {
        worker_id.clone()
    } else {
        stored_instance_id
    };
    let registration_version = i64_to_u32(row.try_get_by_index(6)?)? as u16;
    let registration =
        serde_json::from_str::<RuntimeWorkerRegistration>(&row.try_get_by_index::<String>(7)?)
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
    let heartbeat_version = i64_to_u32(row.try_get_by_index(8)?)? as u16;
    let available_slots = i64_to_u32(row.try_get_by_index(9)?)?;
    let details =
        serde_json::from_str::<RuntimeWorkerHeartbeatDetails>(&row.try_get_by_index::<String>(10)?)
            .unwrap_or_else(|_| RuntimeWorkerHeartbeatDetails {
                version: heartbeat_version,
                available_slots,
                active_lease_ids: Vec::new(),
                last_error: None,
                health_json: json!({}),
            });

    Ok(RuntimeWorkerHeartbeat {
        worker_id,
        started_at: i64_to_u64(row.try_get_by_index(1)?)?,
        last_heartbeat_at: i64_to_u64(row.try_get_by_index(2)?)?,
        status: row.try_get_by_index(3)?,
        queue_names,
        instance_id,
        registration,
        details,
        current_job_id: row.try_get_by_index(11)?,
        current_stage_id: row.try_get_by_index(12)?,
        diagnostic_json: parse_json_value(row.try_get_by_index::<String>(13)?, json!({})),
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

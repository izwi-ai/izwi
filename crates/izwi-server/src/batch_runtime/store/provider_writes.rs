//! Reserved provider writes: reserve, acknowledge, publish and clean up.

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct NewProviderWriteReservation {
    pub write_id: String,
    pub tenant_scope: String,
    pub storage_namespace: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub expected_size_bytes: u64,
    pub expected_sha256: String,
    pub lifetime_ms: u64,
    pub provider_request: MediaWriteRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderWriteReservation {
    pub write_id: String,
    pub reservation_token: String,
    pub created_at: u64,
    pub expires_at: u64,
    pub tenant_scope: String,
    pub storage_namespace: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub expected_size_bytes: u64,
    pub expected_sha256: String,
    pub provider_request: MediaWriteRequest,
    pub storage_key: Option<String>,
    pub cleanup_claim_token: Option<String>,
    pub cleanup_attempt_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProviderWriteRequestEnvelope {
    version: u16,
    request: ProviderWriteRequestV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProviderWriteRequestV1 {
    namespace: MediaNamespace,
    record_id: String,
    preferred_filename: Option<String>,
    content_type: String,
    metadata: HookMetadata,
}

impl From<&MediaWriteRequest> for ProviderWriteRequestV1 {
    fn from(request: &MediaWriteRequest) -> Self {
        Self {
            namespace: request.namespace.clone(),
            record_id: request.record_id.clone(),
            preferred_filename: request.preferred_filename.clone(),
            content_type: request.content_type.clone(),
            metadata: request.metadata.clone(),
        }
    }
}

impl From<ProviderWriteRequestV1> for MediaWriteRequest {
    fn from(request: ProviderWriteRequestV1) -> Self {
        Self {
            namespace: request.namespace,
            record_id: request.record_id,
            preferred_filename: request.preferred_filename,
            content_type: request.content_type,
            metadata: request.metadata,
        }
    }
}

pub(super) const MAX_PROVIDER_WRITE_CLEANUP_BATCH: usize = 64;

pub(super) const MAX_PROVIDER_WRITE_NAMESPACE_BYTES: usize = 128;

pub(super) const MAX_PROVIDER_WRITE_CONTENT_TYPE_BYTES: usize = 256;

pub(super) const MAX_PROVIDER_WRITE_FILENAME_BYTES: usize = 1024;

pub(super) const MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES: usize = 8 * 1024;

pub(super) const MAX_PROVIDER_WRITE_RECORD_ID_BYTES: usize = 64;

pub(super) const MAX_PROVIDER_WRITE_METADATA_ENTRIES: usize = 16;

pub(super) const MAX_PROVIDER_WRITE_METADATA_KEY_BYTES: usize = 128;

pub(super) const MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES: usize = 1024;

pub(super) const MAX_PROVIDER_WRITE_METADATA_BYTES: usize = 4 * 1024;

pub(super) const MAX_PROVIDER_WRITE_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) const MAX_PROVIDER_WRITE_LIFETIME_MS: u64 = 10 * 60 * 1000;

pub(super) const PROVIDER_WRITE_CLEANUP_CLAIM_MS: u64 = 30 * 1000;

pub(super) const PROVIDER_WRITE_REQUEST_ENVELOPE_VERSION: u16 = 1;

pub(super) async fn quarantine_invalid_provider_write<C: ConnectionTrait>(
    db: &C,
    write_id: &str,
    now: i64,
    error: &str,
) -> anyhow::Result<()> {
    let available_at = now.saturating_add(i64::try_from(MAX_ARTIFACT_CLEANUP_BACKOFF_MS)?);
    let result = db
        .execute_raw(raw::statement(
            db,
            r#"
            UPDATE provider_write_reservations
            SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                cleanup_attempt_count = CASE
                    WHEN cleanup_attempt_count < 4294967295
                    THEN cleanup_attempt_count + 1
                    ELSE cleanup_attempt_count
                END,
                last_error = ?3, cleanup_claim_token = NULL,
                cleanup_claim_expires_at = NULL
            WHERE write_id = ?4 AND state = 'cleanup_claimed'
            "#,
            vec![
                now.into(),
                available_at.into(),
                truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                write_id.into(),
            ],
        )?)
        .await
        .context("Failed to quarantine invalid provider write request")?;
    anyhow::ensure!(
        result.rows_affected() == 1,
        "Invalid provider write request lost its cleanup claim"
    );
    Ok(())
}

pub(super) fn validate_provider_write_input(input: &NewProviderWriteReservation) -> anyhow::Result<()> {
    let write_id = uuid::Uuid::parse_str(&input.write_id)
        .map_err(|_| anyhow!("Invalid provider write identity"))?;
    anyhow::ensure!(
        write_id.hyphenated().to_string() == input.write_id,
        "Invalid provider write identity"
    );
    validate_artifact_cleanup_tenant(&input.tenant_scope)?;
    anyhow::ensure!(
        !input.storage_namespace.is_empty()
            && input.storage_namespace.len() <= MAX_PROVIDER_WRITE_NAMESPACE_BYTES
            && !input.storage_namespace.chars().any(char::is_control),
        "Invalid provider write namespace"
    );
    anyhow::ensure!(
        !input.content_type.is_empty()
            && input.content_type.len() <= MAX_PROVIDER_WRITE_CONTENT_TYPE_BYTES
            && !input.content_type.chars().any(char::is_control),
        "Invalid provider write content type"
    );
    anyhow::ensure!(
        input.filename.as_ref().is_none_or(|filename| {
            !filename.is_empty()
                && filename.len() <= MAX_PROVIDER_WRITE_FILENAME_BYTES
                && !filename.chars().any(char::is_control)
        }),
        "Invalid provider write filename"
    );
    anyhow::ensure!(
        input.expected_size_bytes > 0 && input.expected_size_bytes <= MAX_PROVIDER_WRITE_BYTES,
        "Invalid provider write size"
    );
    anyhow::ensure!(
        input.expected_sha256.len() == 64
            && input
                .expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "Invalid provider write digest"
    );
    anyhow::ensure!(
        input.lifetime_ms > 0 && input.lifetime_ms <= MAX_PROVIDER_WRITE_LIFETIME_MS,
        "Invalid provider write lifetime"
    );
    validate_provider_write_request(
        &input.provider_request,
        &input.write_id,
        &input.tenant_scope,
        &input.storage_namespace,
        &input.content_type,
        input.filename.as_deref(),
    )?;
    Ok(())
}

pub(super) fn validate_provider_write_request(
    request: &MediaWriteRequest,
    write_id: &str,
    tenant_scope: &str,
    storage_namespace: &str,
    content_type: &str,
    filename: Option<&str>,
) -> anyhow::Result<()> {
    match &request.namespace {
        MediaNamespace::Other(namespace) => anyhow::ensure!(
            !namespace.is_empty()
                && namespace.len() <= MAX_PROVIDER_WRITE_NAMESPACE_BYTES
                && !namespace.chars().any(char::is_control),
            "Invalid provider request namespace"
        ),
        MediaNamespace::TranscriptionUpload
        | MediaNamespace::DiarizationUpload
        | MediaNamespace::GeneratedSpeech
        | MediaNamespace::SavedVoice
        | MediaNamespace::ChatMedia
        | MediaNamespace::Export => {}
    }
    anyhow::ensure!(
        provider_namespace_storage_value(&request.namespace) == storage_namespace,
        "Provider request namespace does not match its reservation"
    );
    anyhow::ensure!(
        request.record_id == write_id
            && !request.record_id.is_empty()
            && request.record_id.len() <= MAX_PROVIDER_WRITE_RECORD_ID_BYTES
            && !request.record_id.chars().any(char::is_control),
        "Invalid provider request record identity"
    );
    anyhow::ensure!(
        request.content_type == content_type && request.preferred_filename.as_deref() == filename,
        "Provider request does not match its reservation"
    );
    anyhow::ensure!(
        request.metadata.get("tenant_id").map(String::as_str) == Some(tenant_scope),
        "Provider request tenant does not match its reservation"
    );
    anyhow::ensure!(
        request.metadata.len() <= MAX_PROVIDER_WRITE_METADATA_ENTRIES,
        "Provider request metadata entry count exceeds the limit"
    );
    let mut metadata_bytes = 0usize;
    for (key, value) in &request.metadata {
        anyhow::ensure!(
            !key.is_empty()
                && key.len() <= MAX_PROVIDER_WRITE_METADATA_KEY_BYTES
                && !key.chars().any(char::is_control),
            "Invalid provider request metadata key"
        );
        anyhow::ensure!(
            value.len() <= MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES
                && !value.chars().any(char::is_control),
            "Invalid provider request metadata value"
        );
        metadata_bytes = metadata_bytes
            .checked_add(key.len())
            .and_then(|bytes| bytes.checked_add(value.len()))
            .context("Provider request metadata size overflowed")?;
    }
    anyhow::ensure!(
        metadata_bytes <= MAX_PROVIDER_WRITE_METADATA_BYTES,
        "Provider request metadata exceeds the aggregate byte limit"
    );
    Ok(())
}

pub(super) fn serialize_provider_write_request(request: &MediaWriteRequest) -> anyhow::Result<String> {
    let mut output = BoundedJsonBuffer::new(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES);
    let result = serde_json::to_writer(
        &mut output,
        &ProviderWriteRequestEnvelope {
            version: PROVIDER_WRITE_REQUEST_ENVELOPE_VERSION,
            request: request.into(),
        },
    );
    if output.overflowed {
        bail!("Provider write request envelope exceeds the byte limit");
    }
    result.context("Failed to serialize provider write request envelope")?;
    String::from_utf8(output.bytes).context("Provider write request envelope was not UTF-8")
}

pub(super) fn deserialize_provider_write_request(raw: &str) -> anyhow::Result<MediaWriteRequest> {
    anyhow::ensure!(
        raw.len() <= MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES,
        "Stored provider write request envelope exceeds the byte limit"
    );
    let envelope: ProviderWriteRequestEnvelope =
        serde_json::from_str(raw).context("Stored provider write request envelope is invalid")?;
    anyhow::ensure!(
        envelope.version == PROVIDER_WRITE_REQUEST_ENVELOPE_VERSION,
        "Stored provider write request envelope version is unsupported"
    );
    Ok(envelope.request.into())
}

pub(super) fn legacy_provider_write_request(
    write_id: &str,
    tenant_scope: &str,
    storage_namespace: &str,
    content_type: &str,
    filename: Option<&str>,
) -> anyhow::Result<MediaWriteRequest> {
    anyhow::ensure!(
        storage_namespace == "artifact-store",
        "Legacy provider write request cannot be reconstructed safely"
    );
    let mut metadata = HookMetadata::new();
    metadata.insert("tenant_id".to_string(), tenant_scope.to_string());
    Ok(MediaWriteRequest {
        namespace: MediaNamespace::Other(storage_namespace.to_string()),
        record_id: write_id.to_string(),
        preferred_filename: filename.map(str::to_string),
        content_type: content_type.to_string(),
        metadata,
    })
}

pub(super) fn provider_namespace_storage_value(namespace: &MediaNamespace) -> &str {
    match namespace {
        MediaNamespace::TranscriptionUpload => "transcription_upload",
        MediaNamespace::DiarizationUpload => "diarization_upload",
        MediaNamespace::GeneratedSpeech => "generated_speech",
        MediaNamespace::SavedVoice => "saved_voice",
        MediaNamespace::ChatMedia => "chat_media",
        MediaNamespace::Export => "export",
        MediaNamespace::Other(namespace) => namespace,
    }
}

pub(super) fn provider_write_request_length_sql(backend: DbBackend) -> anyhow::Result<&'static str> {
    match backend {
        DbBackend::Sqlite => Ok("LENGTH(CAST(provider_request_json AS BLOB))"),
        DbBackend::Postgres => Ok("OCTET_LENGTH(provider_request_json)"),
        backend => bail!("Unsupported provider write database backend: {backend:?}"),
    }
}

pub(super) struct BoundedJsonBuffer {
    bytes: Vec<u8>,
    max_bytes: usize,
    overflowed: bool,
}

impl BoundedJsonBuffer {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(max_bytes.min(1024)),
            max_bytes,
            overflowed: false,
        }
    }
}

impl Write for BoundedJsonBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(bytes.len()) else {
            self.overflowed = true;
            return Err(io::Error::other("bounded JSON buffer overflow"));
        };
        if new_len > self.max_bytes {
            self.overflowed = true;
            return Err(io::Error::other("bounded JSON buffer limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) async fn get_provider_write_with<C: ConnectionTrait>(
    db: &C,
    write_id: &str,
) -> anyhow::Result<Option<ProviderWriteReservation>> {
    let request_length = provider_write_request_length_sql(db.get_database_backend())?;
    let sql = format!(
        "{} WHERE write_id = ?1 AND (provider_request_json IS NULL OR {request_length} <= ?2)",
        select_provider_write_reservations!()
    );
    let row = db
        .query_one_raw(raw::statement(
            db,
            sql,
            vec![
                write_id.into(),
                i64::try_from(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES)?.into(),
            ],
        )?)
        .await?;
    row.as_ref().map(map_provider_write).transpose()
}

pub(super) async fn mark_provider_write_cleanup_pending_with<C: ConnectionTrait>(
    db: &C,
    reservation: &ProviderWriteReservation,
    now: i64,
    error: &str,
) -> anyhow::Result<bool> {
    let available_at = i64::try_from(reservation.expires_at)?.max(now);
    let result = db
        .execute_raw(raw::statement(
            db,
            r#"
            UPDATE provider_write_reservations
            SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                last_error = ?3, cleanup_claim_token = NULL,
                cleanup_claim_expires_at = NULL
            WHERE write_id = ?4 AND reservation_token = ?5
              AND state IN ('reserved', 'stored')
            "#,
            vec![
                now.into(),
                available_at.into(),
                truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                reservation.write_id.clone().into(),
                reservation.reservation_token.clone().into(),
            ],
        )?)
        .await?;
    Ok(result.rows_affected() == 1)
}

impl BatchRuntimeStore {
    pub(crate) async fn reserve_provider_write(
        &self,
        input: NewProviderWriteReservation,
    ) -> anyhow::Result<ProviderWriteReservation> {
        validate_provider_write_input(&input)?;
        let request_envelope_json = serialize_provider_write_request(&input.provider_request)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start provider write reservation transaction")?;
        let limits = self.provider_write_limits();
        if !acquire_quota_with(&tx, QuotaKind::ProviderWrite, &input.tenant_scope, limits).await? {
            tx.rollback().await?;
            bail!(
                "Provider write reservation capacity exhausted ({} per tenant, {} in total)",
                limits.per_tenant,
                limits.global
            );
        }
        let now = self.now_millis();
        let expires_at = now.saturating_add(i64::try_from(input.lifetime_ms)?);
        let write_id = input.write_id.clone();
        let reservation_token = new_uuid();
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO provider_write_reservations (
                write_id, reservation_token, created_at, updated_at, expires_at,
                available_at, state, tenant_scope, storage_namespace,
                content_type, filename, expected_size_bytes, expected_sha256,
                provider_request_json,
                storage_key, cleanup_claim_token, cleanup_claim_expires_at,
                cleanup_attempt_count, last_error
            ) VALUES (?1, ?2, ?3, ?3, ?4, ?4, 'reserved', ?5, ?6, ?7, ?8,
                      ?9, ?10, ?11, NULL, NULL, NULL, 0, NULL)
            "#,
            vec![
                write_id.clone().into(),
                reservation_token.clone().into(),
                now.into(),
                expires_at.into(),
                input.tenant_scope.into(),
                input.storage_namespace.into(),
                input.content_type.into(),
                opt_string(input.filename),
                u64_to_i64_value(input.expected_size_bytes)?,
                input.expected_sha256.into(),
                request_envelope_json.into(),
            ],
        )?)
        .await
        .context("Failed to reserve provider write")?;
        let reservation = get_provider_write_with(&tx, &write_id)
            .await?
            .context("Reserved provider write was not found")?;
        tx.commit().await?;
        Ok(reservation)
    }

    pub(crate) async fn record_provider_write_stored(
        &self,
        reservation: &ProviderWriteReservation,
        storage_key: &str,
    ) -> anyhow::Result<Option<ProviderWriteReservation>> {
        validate_artifact_cleanup_storage_key(storage_key)?;
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE provider_write_reservations
                SET state = 'stored', storage_key = ?1, updated_at = ?2
                WHERE write_id = ?3 AND reservation_token = ?4
                  AND state = 'reserved' AND expires_at > ?2
                "#,
                vec![
                    storage_key.into(),
                    now.into(),
                    reservation.write_id.clone().into(),
                    reservation.reservation_token.clone().into(),
                ],
            )?)
            .await?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }
        get_provider_write_with(db, &reservation.write_id).await
    }

    pub(crate) async fn abandon_provider_write(
        &self,
        reservation: &ProviderWriteReservation,
        error: &str,
    ) -> anyhow::Result<bool> {
        let now = self.now_millis();
        let available_at = i64::try_from(reservation.expires_at)?.max(now);
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE provider_write_reservations
                SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                    last_error = ?3, cleanup_claim_token = NULL,
                    cleanup_claim_expires_at = NULL
                WHERE write_id = ?4 AND reservation_token = ?5
                  AND state IN ('reserved', 'stored')
                "#,
                vec![
                    now.into(),
                    available_at.into(),
                    truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                    reservation.write_id.clone().into(),
                    reservation.reservation_token.clone().into(),
                ],
            )?)
            .await?;
        Ok(result.rows_affected() == 1)
    }

    pub(crate) async fn publish_reserved_opaque_artifact(
        &self,
        reservation: &ProviderWriteReservation,
        input: NewMediaAsset,
    ) -> anyhow::Result<Option<MediaAsset>> {
        anyhow::ensure!(
            input.storage_key.as_str() == reservation.storage_key.as_deref().unwrap_or_default()
                && input.content_type == reservation.content_type
                && input.filename == reservation.filename
                && input.size_bytes == reservation.expected_size_bytes
                && input.sha256.as_deref() == Some(reservation.expected_sha256.as_str()),
            "Provider write publication did not match its reservation"
        );
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let now = self.now_millis();
        let row = get_provider_write_with(&tx, &reservation.write_id).await?;
        if row.as_ref() != Some(reservation) {
            tx.rollback().await?;
            return Ok(None);
        }
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO media_assets (
                id, created_at, updated_at, asset_kind, storage_namespace,
                storage_key, content_type, filename, size_bytes, sha256,
                duration_secs, sample_rate_hz, channel_count, peak_amplitude,
                rms_amplitude, source_asset_id, canonical_profile_version,
                scan_status, retention_policy, deleted_at, metadata_json
            ) VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                      ?12, ?13, ?14, ?15, ?16, ?17, ?18, NULL, ?19)
        "#,
            vec![
                id.clone().into(),
                now.into(),
                input.asset_kind.into(),
                input.storage_namespace.into(),
                input.storage_key.into(),
                input.content_type.into(),
                opt_string(input.filename),
                u64_to_i64_value(input.size_bytes)?,
                opt_string(input.sha256),
                opt_f64(input.duration_secs),
                opt_u32(input.sample_rate_hz),
                opt_u16(input.channel_count),
                opt_f32(input.peak_amplitude),
                opt_f32(input.rms_amplitude),
                opt_string(input.source_asset_id),
                opt_string(input.canonical_profile_version),
                input.scan_status.into(),
                input.retention_policy.into(),
                metadata_json.into(),
            ],
        )?)
        .await?;
        let deleted = tx.execute_raw(raw::statement(&tx,
            "DELETE FROM provider_write_reservations WHERE write_id = ?1 AND reservation_token = ?2 AND state = 'stored'",
            vec![reservation.write_id.clone().into(), reservation.reservation_token.clone().into()],
        )?).await?;
        if deleted.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        release_quota_with(&tx, QuotaKind::ProviderWrite, &reservation.tenant_scope, 1).await?;
        let asset = get_media_asset_with(&tx, &id).await?;
        tx.commit().await?;
        Ok(asset)
    }

    /// Atomically consume a stored provider reservation, create its opaque media
    /// row, and attach that row to the exact active stage attempt.
    ///
    /// `Existing` means the attempt already committed the same publication key;
    /// the duplicate reservation is atomically retained for bounded provider
    /// recovery. `None` means the attempt fence was lost and its reservation was
    /// likewise retained for recovery.
    pub(crate) async fn publish_reserved_opaque_stage_artifact(
        &self,
        reservation: &ProviderWriteReservation,
        lease: &StageLease,
        media: NewMediaAsset,
        mut artifact: NewStageOutputArtifact,
        progress: serde_json::Value,
    ) -> anyhow::Result<Option<ReservedStageArtifactPublication>> {
        anyhow::ensure!(
            media.storage_key.as_str() == reservation.storage_key.as_deref().unwrap_or_default()
                && media.content_type == reservation.content_type
                && media.filename == reservation.filename
                && media.size_bytes == reservation.expected_size_bytes
                && media.sha256.as_deref() == Some(reservation.expected_sha256.as_str()),
            "Provider write publication did not match its reservation"
        );
        anyhow::ensure!(
            artifact.media_asset_id.is_none()
                && artifact.text_asset_id.is_none()
                && artifact.storage_key.is_none(),
            "Opaque stage artifact identity is assigned by the metadata transaction"
        );
        if !matches!(
            artifact.artifact_role,
            RuntimeArtifactRole::OutputPrimary
                | RuntimeArtifactRole::OutputIntermediate
                | RuntimeArtifactRole::Debug
        ) {
            bail!("Attempt-owned artifact publication requires an output or debug role");
        }
        let publication_key = artifact.publication_key.trim().to_string();
        if publication_key.is_empty() {
            bail!("Attempt-owned artifact publication requires a publication key");
        }
        let Some(attempt_token) = lease.attempt_token.as_deref() else {
            return Ok(None);
        };
        let is_speech_pcm = publication_key.starts_with("speech-pcm/");
        artifact.publication_key = publication_key.clone();
        artifact.content_type = Some(media.content_type.clone());
        artifact.filename = media.filename.clone();
        artifact.size_bytes = Some(media.size_bytes);
        artifact.sha256 = media.sha256.clone();
        let expected_media = media.clone();
        let expected_artifact = artifact.clone();
        let pcm_replay_entry_limit = self.pcm_replay_entry_limit();
        anyhow::ensure!(
            pcm_replay_entry_limit > 0,
            "Speech PCM replay entry limit must be positive"
        );
        let progress_json = bounded_json_string(
            &progress,
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Stage progress",
        )?;

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start opaque stage artifact publication transaction")?;
        lock_stage_job_with(&tx, &lease.stage_id).await?;
        let now = self.now_millis();
        let row = get_provider_write_with(&tx, &reservation.write_id).await?;
        if row.as_ref() != Some(reservation) {
            tx.rollback().await?;
            return Ok(None);
        }
        let lock_clause = match tx.get_database_backend() {
            DbBackend::Sqlite => "",
            DbBackend::Postgres => " FOR UPDATE OF s, j",
            backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
        };
        let ownership_sql = format!(
            r#"
            SELECT s.id
            FROM job_stages s
            JOIN runtime_jobs j ON j.id = s.job_id
            WHERE s.id = ?1
              AND s.status = 'running'
              AND s.cancellation_state IS NULL
              AND s.worker_id = ?2
              AND s.attempt_count = ?3
              AND s.attempt_token = ?4
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?5
              AND j.status IN ('queued', 'running', 'retrying')
              AND j.cancellation_state IS NULL
              AND COALESCE(j.admission_tenant, 'anonymous') = ?6
            LIMIT 1{lock_clause}
            "#
        );
        let owns_stage = tx
            .query_one_raw(raw::statement(
                &tx,
                ownership_sql,
                vec![
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.into(),
                    now.into(),
                    reservation.tenant_scope.clone().into(),
                ],
            )?)
            .await
            .context("Failed to lock opaque artifact stage ownership")?
            .is_some();
        if !owns_stage {
            if !mark_provider_write_cleanup_pending_with(
                &tx,
                reservation,
                now,
                "Stage attempt lost ownership before opaque artifact publication",
            )
            .await?
            {
                tx.rollback().await?;
                return Ok(None);
            }
            tx.commit().await?;
            return Ok(None);
        }
        if let Some(existing) =
            active_stage_output_for_key_with(&tx, lease, attempt_token, &publication_key, now)
                .await?
        {
            let exact = match existing.media_asset_id.as_deref() {
                Some(id) => get_media_asset_with(&tx, id).await?.is_some_and(|asset| {
                    reserved_stage_artifact_matches(
                        &existing,
                        &asset,
                        &expected_media,
                        &expected_artifact,
                    )
                }),
                None => false,
            };
            if !mark_provider_write_cleanup_pending_with(
                &tx,
                reservation,
                now,
                "Attempt publication key was already committed",
            )
            .await?
            {
                tx.rollback().await?;
                return Ok(None);
            }
            tx.commit().await?;
            anyhow::ensure!(
                exact,
                "Attempt publication key was reused for a different opaque artifact"
            );
            return Ok(Some(ReservedStageArtifactPublication::Existing(existing)));
        }

        let media_id = new_uuid();
        let media_metadata_json = json_to_db_string(&media.metadata_json, "{}")?;
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO media_assets (
                id, created_at, updated_at, asset_kind, storage_namespace,
                storage_key, content_type, filename, size_bytes, sha256,
                duration_secs, sample_rate_hz, channel_count, peak_amplitude,
                rms_amplitude, source_asset_id, canonical_profile_version,
                scan_status, retention_policy, deleted_at, metadata_json
            ) VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                      ?12, ?13, ?14, ?15, ?16, ?17, ?18, NULL, ?19)
            "#,
            vec![
                media_id.clone().into(),
                now.into(),
                media.asset_kind.into(),
                media.storage_namespace.into(),
                media.storage_key.into(),
                media.content_type.into(),
                opt_string(media.filename),
                u64_to_i64_value(media.size_bytes)?,
                opt_string(media.sha256),
                opt_f64(media.duration_secs),
                opt_u32(media.sample_rate_hz),
                opt_u16(media.channel_count),
                opt_f32(media.peak_amplitude),
                opt_f32(media.rms_amplitude),
                opt_string(media.source_asset_id),
                opt_string(media.canonical_profile_version),
                media.scan_status.into(),
                media.retention_policy.into(),
                media_metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to create reserved opaque media asset")?;

        artifact.media_asset_id = Some(media_id.clone());
        let artifact_id = new_uuid();
        let artifact_metadata_json = json_to_db_string(&artifact.metadata_json, "{}")?;
        let conflict_clause = match tx.get_database_backend() {
            DbBackend::Sqlite | DbBackend::Postgres => {
                "ON CONFLICT(stage_id, producer_attempt_token, publication_key) DO NOTHING"
            }
            backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
        };
        let insert_sql = format!(
            r#"
            INSERT INTO runtime_artifacts (
                id, job_id, stage_id, producer_attempt_count,
                producer_attempt_token, publication_key, created_at,
                artifact_kind, artifact_role, media_asset_id, text_asset_id,
                storage_key, content_type, filename, size_bytes, sha256,
                metadata_json, retention_policy
            )
            SELECT
                ?1, s.job_id, s.id, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                ?10, ?11, ?12, ?13, ?14, ?15, ?16
            FROM job_stages s
            JOIN runtime_jobs j ON j.id = s.job_id
            WHERE s.id = ?17
              AND s.status = 'running'
              AND s.cancellation_state IS NULL
              AND s.worker_id = ?18
              AND s.attempt_count = ?2
              AND s.attempt_token = ?3
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?5
              AND j.status IN ('queued', 'running', 'retrying')
              AND j.cancellation_state IS NULL
              AND COALESCE(j.admission_tenant, 'anonymous') = ?20
              AND (
                  ?4 NOT LIKE 'speech-pcm/%'
                  OR (
                      SELECT COUNT(*) FROM runtime_artifacts existing_pcm
                      WHERE existing_pcm.job_id = s.job_id
                        AND existing_pcm.publication_key >= 'speech-pcm/'
                        AND existing_pcm.publication_key < 'speech-pcm0'
                  ) < ?19
              )
            {conflict_clause}
            "#
        );
        tx.execute_raw(raw::statement(
            &tx,
            insert_sql,
            vec![
                artifact_id.into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.into(),
                publication_key.clone().into(),
                now.into(),
                artifact.artifact_kind.as_db_value().into(),
                artifact.artifact_role.as_db_value().into(),
                media_id.clone().into(),
                opt_string(artifact.text_asset_id),
                opt_string(artifact.storage_key),
                opt_string(artifact.content_type),
                opt_string(artifact.filename),
                opt_u64(artifact.size_bytes),
                opt_string(artifact.sha256),
                artifact_metadata_json.into(),
                artifact.retention_policy.into(),
                lease.stage_id.clone().into(),
                lease.worker_id.clone().into(),
                i64::try_from(pcm_replay_entry_limit)?.into(),
                reservation.tenant_scope.clone().into(),
            ],
        )?)
        .await
        .context("Failed to attach reserved opaque artifact to stage attempt")?;

        let published =
            stage_output_for_key_with(&tx, lease, attempt_token, &publication_key).await?;
        let Some(published) = published else {
            tx.execute_raw(raw::statement(
                &tx,
                "DELETE FROM media_assets WHERE id = ?1",
                vec![media_id.into()],
            )?)
            .await?;
            if !mark_provider_write_cleanup_pending_with(
                &tx,
                reservation,
                now,
                "Stage attempt lost ownership before opaque artifact publication",
            )
            .await?
            {
                tx.rollback().await?;
                return Ok(None);
            }
            tx.commit().await?;
            if is_speech_pcm {
                bail!("speech_storage_limit: replay entry quota exceeded");
            }
            return Ok(None);
        };
        if published.media_asset_id.as_deref() != Some(media_id.as_str()) {
            tx.execute_raw(raw::statement(
                &tx,
                "DELETE FROM media_assets WHERE id = ?1",
                vec![media_id.into()],
            )?)
            .await?;
            let still_active =
                active_stage_output_for_key_with(&tx, lease, attempt_token, &publication_key, now)
                    .await?
                    .is_some_and(|artifact| artifact.id == published.id);
            let exact = match published.media_asset_id.as_deref() {
                Some(id) => get_media_asset_with(&tx, id).await?.is_some_and(|asset| {
                    reserved_stage_artifact_matches(
                        &published,
                        &asset,
                        &expected_media,
                        &expected_artifact,
                    )
                }),
                None => false,
            };
            if !mark_provider_write_cleanup_pending_with(
                &tx,
                reservation,
                now,
                "Concurrent attempt publication won the opaque artifact key",
            )
            .await?
            {
                tx.rollback().await?;
                return Ok(None);
            }
            tx.commit().await?;
            anyhow::ensure!(
                still_active && exact,
                "Attempt publication key was reused for a different opaque artifact"
            );
            return Ok(Some(ReservedStageArtifactPublication::Existing(published)));
        }
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE job_stages
            SET progress_json = ?1, updated_at = ?2
            WHERE id = ?3
              AND worker_id = ?4
              AND attempt_count = ?5
              AND attempt_token = ?6
            "#,
            vec![
                progress_json.into(),
                now.into(),
                lease.stage_id.clone().into(),
                lease.worker_id.clone().into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.into(),
            ],
        )?)
        .await
        .context("Failed to checkpoint opaque stage artifact publication")?;
        let deleted = tx.execute_raw(raw::statement(&tx,
            "DELETE FROM provider_write_reservations WHERE write_id = ?1 AND reservation_token = ?2 AND state = 'stored'",
            vec![reservation.write_id.clone().into(), reservation.reservation_token.clone().into()],
        )?).await?;
        if deleted.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        release_quota_with(&tx, QuotaKind::ProviderWrite, &reservation.tenant_scope, 1).await?;
        let asset = get_media_asset_with(&tx, &media_id)
            .await?
            .context("Published opaque stage media asset was not found")?;
        tx.commit()
            .await
            .context("Failed to commit opaque stage artifact publication")?;
        Ok(Some(ReservedStageArtifactPublication::Published {
            asset: Box::new(asset),
            artifact: published,
        }))
    }

    pub(crate) async fn claim_due_provider_write_cleanup(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<ProviderWriteReservation>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let now = self.now_millis();
        let request_length = provider_write_request_length_sql(tx.get_database_backend())?;
        let selection_sql = format!(
            r#"
            SELECT write_id FROM provider_write_reservations
            WHERE (
                (state IN ('reserved', 'stored', 'cleanup_pending') AND available_at <= ?1)
                OR (state = 'cleanup_claimed' AND cleanup_claim_expires_at <= ?1)
            )
              AND (provider_request_json IS NOT NULL OR storage_namespace = 'artifact-store')
              AND (provider_request_json IS NULL OR {request_length} <= ?2)
            ORDER BY available_at ASC, created_at ASC, write_id ASC LIMIT ?3
            "#
        );
        let rows = tx
            .query_all_raw(raw::statement(
                &tx,
                selection_sql,
                vec![
                    now.into(),
                    i64::try_from(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES)?.into(),
                    i64::try_from(limit.min(MAX_PROVIDER_WRITE_CLEANUP_BATCH))?.into(),
                ],
            )?)
            .await?;
        let mut claimed = Vec::with_capacity(rows.len());
        for row in rows {
            let write_id: String = row.try_get_by_index(0)?;
            let claim_token = new_uuid();
            let claim_expires_at =
                now.saturating_add(i64::try_from(PROVIDER_WRITE_CLEANUP_CLAIM_MS)?);
            let result = tx
                .execute_raw(raw::statement(
                    &tx,
                    r#"
                UPDATE provider_write_reservations
                SET state = 'cleanup_claimed', updated_at = ?1,
                    cleanup_claim_token = ?2, cleanup_claim_expires_at = ?3
                WHERE write_id = ?4 AND (
                    (state IN ('reserved', 'stored', 'cleanup_pending') AND available_at <= ?1)
                    OR (state = 'cleanup_claimed' AND cleanup_claim_expires_at <= ?1)
                )
            "#,
                    vec![
                        now.into(),
                        claim_token.into(),
                        claim_expires_at.into(),
                        write_id.clone().into(),
                    ],
                )?)
                .await?;
            if result.rows_affected() == 1 {
                match get_provider_write_with(&tx, &write_id).await {
                    Ok(Some(reservation)) => claimed.push(reservation),
                    Ok(None) => {
                        quarantine_invalid_provider_write(
                            &tx,
                            &write_id,
                            now,
                            "Provider write request envelope could not be loaded safely",
                        )
                        .await?;
                    }
                    Err(error) => {
                        quarantine_invalid_provider_write(&tx, &write_id, now, &error.to_string())
                            .await?;
                    }
                }
            }
        }
        tx.commit().await?;
        Ok(claimed)
    }

    pub(crate) async fn complete_provider_write_cleanup(
        &self,
        reservation: &ProviderWriteReservation,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let result = tx.execute_raw(raw::statement(&tx,
            "DELETE FROM provider_write_reservations WHERE write_id = ?1 AND state = 'cleanup_claimed' AND cleanup_claim_token = ?2",
            vec![reservation.write_id.clone().into(), opt_string(reservation.cleanup_claim_token.clone())],
        )?).await?;
        let completed = result.rows_affected() == 1;
        if completed {
            release_quota_with(&tx, QuotaKind::ProviderWrite, &reservation.tenant_scope, 1).await?;
        }
        tx.commit().await?;
        Ok(completed)
    }

    pub(crate) async fn defer_provider_write_cleanup(
        &self,
        reservation: &ProviderWriteReservation,
        error: &str,
    ) -> anyhow::Result<bool> {
        let attempt = reservation.cleanup_attempt_count.saturating_add(1);
        let backoff = 1_000_u64
            .checked_shl(reservation.cleanup_attempt_count.min(12))
            .unwrap_or(MAX_ARTIFACT_CLEANUP_BACKOFF_MS)
            .min(MAX_ARTIFACT_CLEANUP_BACKOFF_MS);
        let now = self.now_millis();
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
            UPDATE provider_write_reservations
            SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                cleanup_attempt_count = ?3, last_error = ?4,
                cleanup_claim_token = NULL, cleanup_claim_expires_at = NULL
            WHERE write_id = ?5 AND state = 'cleanup_claimed' AND cleanup_claim_token = ?6
        "#,
                vec![
                    now.into(),
                    now.saturating_add(i64::try_from(backoff)?).into(),
                    i64::from(attempt).into(),
                    truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                    reservation.write_id.clone().into(),
                    opt_string(reservation.cleanup_claim_token.clone()),
                ],
            )?)
            .await?;
        Ok(result.rows_affected() == 1)
    }

    pub(super) fn provider_write_limits(&self) -> QuotaLimits {
        #[cfg(test)]
        if let Some(limits) = self.test_provider_write_limits {
            return limits;
        }
        crate::env_config::tuning().provider_write_limits
    }
}

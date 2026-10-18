//! Atomic durable acceptance of a whole job graph.

use super::*;

/// The bounded, single-stage durable graph used by the first text-only TTS
/// acceptance path. Reference media is deliberately excluded until object
/// publication has its own crash-safe ownership ledger.
#[derive(Debug, Clone)]
pub struct NewDurableTextTtsAcceptance {
    pub projection: NewSpeechHistoryRecord,
    pub request_json: serde_json::Value,
    pub model_snapshot_json: serde_json::Value,
    pub retry_policy_json: serde_json::Value,
    pub priority: i32,
    pub max_attempts: u32,
    pub correlation_id: Option<String>,
    pub stage_kind: String,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
    pub reservation: Option<DurableIdempotencyReservation>,
    pub idempotency_retention_ms: u64,
}

#[derive(Debug, Clone)]
pub struct DurableTextTtsAcceptance {
    pub record: SpeechHistoryRecord,
    pub job: RuntimeJob,
    pub stage: JobStage,
    pub input_artifact: RuntimeArtifact,
}

// The committed payload is the common case and flows straight into the
// durable acceptance record; boxing it would heap-allocate every commit to
// slim the exceptional no-data variant.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum DurableTextTtsAcceptanceOutcome {
    Committed(DurableTextTtsAcceptance),
    ReservationLost,
}

/// One queued job graph accepted in a single write transaction: an optional
/// text input, the job, its input artifacts, every stage, and, when keyed, the
/// idempotency result. The caller names the text input and artifacts so that
/// artifact metadata and stage inputs can reference them before they exist.
#[derive(Debug, Clone)]
pub struct NewDurableJobAcceptance {
    pub job: NewRuntimeJob,
    pub text_input: Option<NewDurableTextInput>,
    pub artifacts: Vec<NewDurableJobArtifact>,
    pub stages: Vec<NewDurableJobStage>,
    pub reservation: Option<DurableIdempotencyReservation>,
    /// The bounded replay payload committed with a keyed acceptance.
    pub idempotency_response: serde_json::Value,
    pub idempotency_retention_ms: u64,
}

#[derive(Debug, Clone)]
pub struct NewDurableTextInput {
    pub id: String,
    pub asset: NewTextAsset,
}

#[derive(Debug, Clone)]
pub struct NewDurableJobArtifact {
    pub id: String,
    pub artifact_kind: RuntimeArtifactKind,
    pub artifact_role: RuntimeArtifactRole,
    pub media_asset_id: Option<String>,
    pub text_asset_id: Option<String>,
    pub storage_key: Option<String>,
    pub content_type: Option<String>,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub metadata_json: serde_json::Value,
    pub retention_policy: String,
}

#[derive(Debug, Clone)]
pub struct NewDurableJobStage {
    pub sequence: u32,
    pub stage_kind: String,
    pub capability: Option<String>,
    pub model_id: Option<String>,
    pub max_attempts: u32,
    pub input_artifact_ids: Vec<String>,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
}

#[derive(Debug, Clone)]
pub struct DurableJobAcceptance {
    pub job: RuntimeJob,
    pub artifacts: Vec<RuntimeArtifact>,
    pub stages: Vec<JobStage>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum DurableJobAcceptanceOutcome {
    Committed(DurableJobAcceptance),
    ReservationLost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DurableAcceptanceFailpoint {
    Projection,
    TextAsset,
    Job,
    Artifact,
    Stage,
    Idempotency,
}

pub const MAX_DURABLE_TTS_TEXT_BYTES: usize = 1024 * 1024;

pub(super) const MAX_DURABLE_TTS_REQUEST_JSON_BYTES: usize = 2 * 1024 * 1024;

pub(super) const MAX_DURABLE_TTS_METADATA_JSON_BYTES: usize = 64 * 1024;

pub(super) const MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES: usize = 64 * 1024;

pub(super) const MAX_DURABLE_TTS_CORRELATION_BYTES: usize = 256;

pub(super) fn validate_acceptance_reservation(
    reservation: &DurableIdempotencyReservation,
    retention_ms: u64,
) -> anyhow::Result<()> {
    validate_durable_idempotency_identity(
        &reservation.tenant_scope,
        &reservation.operation,
        &reservation.idempotency_key,
        reservation.digest_version,
        &reservation.request_digest,
    )?;
    validate_bounded_field("reservation token", &reservation.reservation_token, 64)?;
    anyhow::ensure!(
        (1..=MAX_DURABLE_IDEMPOTENCY_RETENTION_MS).contains(&retention_ms),
        "Durable idempotency retention must be between 1 and {MAX_DURABLE_IDEMPOTENCY_RETENTION_MS} milliseconds"
    );
    Ok(())
}

impl BatchRuntimeStore {
    /// Atomically publish one text-only TTS job graph and, when supplied, its
    /// durable idempotency result. No worker can observe the queued stage
    /// before the projection, input, job, and replay record are all committed.
    pub async fn accept_durable_text_tts(
        &self,
        input: NewDurableTextTtsAcceptance,
    ) -> anyhow::Result<DurableTextTtsAcceptanceOutcome> {
        anyhow::ensure!(
            input.projection.route_kind == SpeechRouteKind::TextToSpeech,
            StoreError::invalid("Durable text TTS acceptance requires a text-to-speech projection")
        );
        anyhow::ensure!(
            input.projection.processing_status == SpeechHistoryProcessingStatus::Pending
                && input.projection.processing_error.is_none(),
            StoreError::invalid("Durable text TTS acceptance requires a clean pending projection")
        );
        anyhow::ensure!(
            input.projection.audio_bytes.is_empty()
                && input.projection.reference_text.is_none()
                && input.projection.saved_voice_id.is_none(),
            StoreError::invalid(
                "Durable text TTS acceptance does not accept reference or output media"
            )
        );
        anyhow::ensure!(
            input.projection.generation_time_ms == 0.0
                && input.projection.audio_duration_secs.is_none()
                && input.projection.rtf.is_none()
                && input.projection.tokens_generated.is_none(),
            StoreError::invalid("Durable text TTS acceptance requires an unexecuted projection")
        );
        anyhow::ensure!(
            input.max_attempts > 0,
            StoreError::invalid("Durable text TTS acceptance requires at least one attempt")
        );
        anyhow::ensure!(
            input.queue_class == QueueClass::BatchTts,
            StoreError::invalid("Durable text TTS acceptance requires the batch TTS queue")
        );
        validate_bounded_field("stage kind", &input.stage_kind, 128)?;

        anyhow::ensure!(
            !input.projection.input_text.is_empty()
                && input.projection.input_text.len() <= MAX_DURABLE_TTS_TEXT_BYTES,
            StoreError::too_large(format!(
                "Durable text TTS input must be between 1 and {MAX_DURABLE_TTS_TEXT_BYTES} bytes"
            ))
        );
        for (name, value) in [
            ("model ID", input.projection.model_id.as_deref()),
            ("speaker", input.projection.speaker.as_deref()),
            ("language", input.projection.language.as_deref()),
            (
                "voice description",
                input.projection.voice_description.as_deref(),
            ),
            ("audio filename", input.projection.audio_filename.as_deref()),
        ] {
            validate_optional_field_bytes(name, value, MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES)?;
        }
        validate_optional_field_bytes(
            "correlation ID",
            input.correlation_id.as_deref(),
            MAX_DURABLE_TTS_CORRELATION_BYTES,
        )?;
        anyhow::ensure!(
            input.projection.audio_mime_type.len() <= MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES,
            StoreError::invalid(format!(
                "Durable text TTS audio MIME type exceeds {MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES} bytes"
            ))
        );

        let model_id = sanitize_optional_text(input.projection.model_id.as_deref(), 160)
            .ok_or_else(|| {
                StoreError::invalid("Durable text TTS acceptance requires a model ID")
            })?;
        let speaker = sanitize_optional_text(input.projection.speaker.as_deref(), 120);
        let language = sanitize_optional_text(input.projection.language.as_deref(), 80);
        let voice_description =
            sanitize_optional_text(input.projection.voice_description.as_deref(), 2_000);
        let input_text = input.projection.input_text.trim().to_string();
        anyhow::ensure!(
            !input_text.is_empty(),
            StoreError::invalid("Durable text TTS acceptance requires non-empty input text")
        );
        let speed = input
            .projection
            .speed
            .filter(|value| value.is_finite() && *value > 0.0);
        let audio_mime_type = sanitize_audio_mime_type(&input.projection.audio_mime_type);
        let audio_filename =
            sanitize_optional_text(input.projection.audio_filename.as_deref(), 260);
        let request_json_string = bounded_json_string(
            &input.request_json,
            MAX_DURABLE_TTS_REQUEST_JSON_BYTES,
            "Durable text TTS request",
        )
        .map_err(|err| StoreError::too_large(err.to_string()))?;
        let model_snapshot_json_string = bounded_json_string(
            &input.model_snapshot_json,
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Durable text TTS model snapshot",
        )
        .map_err(|err| StoreError::invalid(err.to_string()))?;
        let retry_policy_json_string = bounded_json_string(
            &input.retry_policy_json,
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Durable text TTS retry policy",
        )
        .map_err(|err| StoreError::invalid(err.to_string()))?;
        let resource_hints = input.resource_hints.normalized();
        let resource_hints_json = bounded_json_string(
            &json!(resource_hints.clone()),
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Durable text TTS resource hints",
        )
        .map_err(|err| StoreError::invalid(err.to_string()))?;
        let admission_tenant = speech_admission_tenant(&input.request_json)?;

        if let Some(reservation) = input.reservation.as_ref() {
            validate_acceptance_reservation(reservation, input.idempotency_retention_ms)?;
        }

        let now = self.now_millis();
        let now_u64 = nonnegative_timestamp(now)?;
        let record_id = new_uuid();
        let text_asset_id = new_uuid();
        let job_id = new_uuid();
        let artifact_id = new_uuid();
        let stage_id = new_uuid();
        let text_sha256 = sha256_hex(input_text.as_bytes());
        let record = SpeechHistoryRecord {
            stop_reason: None,
            id: record_id.clone(),
            created_at: now_u64,
            route_kind: SpeechRouteKind::TextToSpeech,
            processing_status: SpeechHistoryProcessingStatus::Pending,
            processing_error: None,
            model_id: Some(model_id.clone()),
            speaker: speaker.clone(),
            language: language.clone(),
            saved_voice_id: None,
            speed,
            input_text: input_text.clone(),
            voice_description: voice_description.clone(),
            reference_text: None,
            generation_time_ms: 0.0,
            audio_duration_secs: None,
            rtf: None,
            tokens_generated: None,
            audio_mime_type: audio_mime_type.clone(),
            audio_filename: audio_filename.clone(),
        };
        // Persist only a bounded lookup key. Replays load the route projection
        // by ID, avoiding a second copy of potentially 1 MiB input text.
        let response_json = json!({"record_id": record_id.clone()});
        let response_json_string = input
            .reservation
            .as_ref()
            .map(|_| {
                bounded_json_string(
                    &response_json,
                    MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
                    "Durable idempotency result",
                )
            })
            .transpose()?;
        let structure_json = json_to_db_string(
            &json!({"source": "tts_input", "route_record_id": record_id.clone()}),
            "{}",
        )?;
        let artifact_metadata_json = json_to_db_string(
            &json!({"route_record_id": record_id.clone(), "input_kind": "text"}),
            "{}",
        )?;
        let input_artifact_ids_json = json_to_db_string(&json!([artifact_id.clone()]), "[]")?;

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start durable text TTS acceptance transaction")?;

        if let Some(reservation) = input.reservation.as_ref() {
            if !durable_reservation_is_live_with(&tx, reservation, now_u64).await? {
                tx.rollback().await?;
                return Ok(DurableTextTtsAcceptanceOutcome::ReservationLost);
            }
        }

        self.check_tts_admission(&tx, &admission_tenant).await?;
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO speech_history_records (
                id, created_at, route_kind, processing_status, processing_error,
                runtime_stage_id, runtime_attempt_token, model_id, speaker,
                language, saved_voice_id, speed, input_text, voice_description,
                reference_text, generation_time_ms, audio_duration_secs, rtf,
                tokens_generated, audio_mime_type, audio_filename, audio_storage_path,
                audio_media_asset_id, audio_artifact_tenant
            )
            VALUES (?1, ?2, 'text_to_speech', 'pending', NULL, NULL, NULL, ?3,
                    ?4, ?5, NULL, ?6, ?7, ?8, NULL, 0.0, NULL, NULL, NULL,
                    ?9, ?10, '', NULL, NULL)
            "#,
            vec![
                record_id.clone().into(),
                now.into(),
                model_id.clone().into(),
                opt_string(speaker),
                opt_string(language.clone()),
                opt_f64(speed),
                input_text.clone().into(),
                opt_string(voice_description),
                audio_mime_type.clone().into(),
                opt_string(audio_filename.clone()),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS projection")?;
        self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Projection)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO text_assets (
                id, created_at, updated_at, raw_text, normalized_text,
                language_hint, character_count, sha256, safety_status,
                retention_policy, structure_json
            )
            VALUES (?1, ?2, ?2, ?3, ?3, ?4, ?5, ?6, 'unchecked', 'default', ?7)
            "#,
            vec![
                text_asset_id.clone().into(),
                now.into(),
                input_text.clone().into(),
                opt_string(language),
                u64_to_i64_value(input_text.chars().count() as u64)?,
                text_sha256.clone().into(),
                structure_json.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS input")?;
        self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::TextAsset)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_jobs (
                id, created_at, updated_at, queued_at, started_at, finished_at,
                job_kind, status, priority, model_id, capability,
                route_record_kind, route_record_id, input_media_asset_id,
                input_text_asset_id, request_json, model_snapshot_json,
                progress_json, error_code, error_message, attempt_count,
                max_attempts, retry_policy_json, idempotency_key,
                correlation_id, cancellation_reason, admission_tenant
            )
            VALUES (?1, ?2, ?2, ?2, NULL, NULL, 'tts_speech', 'queued', ?3,
                    ?4, 'tts', 'text_to_speech', ?5, NULL, ?6, ?7, ?8,
                    NULL, NULL, NULL, 0, ?9, ?10, ?11, ?12, NULL, ?13)
            "#,
            vec![
                job_id.clone().into(),
                now.into(),
                input.priority.into(),
                model_id.clone().into(),
                record_id.clone().into(),
                text_asset_id.clone().into(),
                request_json_string.into(),
                model_snapshot_json_string.into(),
                u32_to_i64_value(input.max_attempts).into(),
                retry_policy_json_string.into(),
                opt_string(
                    input
                        .reservation
                        .as_ref()
                        .map(|reservation| reservation.idempotency_key.clone()),
                ),
                opt_string(input.correlation_id),
                admission_tenant.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS job")?;
        self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Job)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_artifacts (
                id, job_id, stage_id, created_at, artifact_kind,
                artifact_role, media_asset_id, text_asset_id, storage_key,
                content_type, filename, size_bytes, sha256, metadata_json,
                retention_policy
            )
            VALUES (?1, ?2, NULL, ?3, 'text', 'input_original', NULL, ?4,
                    NULL, 'text/plain', ?5, ?6, ?7, ?8, 'default')
            "#,
            vec![
                artifact_id.clone().into(),
                job_id.clone().into(),
                now.into(),
                text_asset_id.clone().into(),
                format!("{record_id}.input.txt").into(),
                u64_to_i64_value(input_text.len() as u64)?,
                text_sha256.into(),
                artifact_metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS input artifact")?;
        self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Artifact)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO job_stages (
                id, job_id, created_at, updated_at, sequence, stage_kind,
                queue_class, resource_hints_json, resource_target,
                required_backend, required_device_class,
                min_resource_memory_bytes, resource_concurrency_weight,
                resource_hints_version,
                status, capability, model_id, worker_id, lease_expires_at,
                available_at, attempt_token, attempt_count, max_attempts,
                input_artifact_ids_json, output_artifact_ids_json,
                progress_json, started_at, finished_at, error_code,
                error_message, cancellation_state
            )
            VALUES (?1, ?2, ?3, ?3, 0, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                    ?11, ?15, 'queued', 'tts', ?12, NULL, NULL, ?3, NULL, 0, ?13,
                    ?14, '[]', NULL, NULL, NULL, NULL, NULL, NULL)
            "#,
            vec![
                stage_id.clone().into(),
                job_id.clone().into(),
                now.into(),
                input.stage_kind.into(),
                input.queue_class.as_db_value().into(),
                resource_hints_json.into(),
                resource_hints.target.as_db_value().into(),
                opt_string(
                    resource_hints
                        .backend
                        .map(|backend| backend.as_db_value().to_string()),
                ),
                opt_string(
                    resource_hints
                        .device_class
                        .map(|device| device.as_db_value().to_string()),
                ),
                opt_u64(resource_hints.min_memory_bytes),
                u32_to_i64_value(resource_hints.concurrency_weight).into(),
                model_id.into(),
                u32_to_i64_value(input.max_attempts).into(),
                input_artifact_ids_json.into(),
                i64::from(resource_hints.version).into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS stage")?;
        self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Stage)?;

        if let Some(reservation) = input.reservation.as_ref() {
            let committed = commit_durable_reservation_with(
                &tx,
                reservation,
                now_u64,
                input.idempotency_retention_ms,
                &job_id,
                response_json_string
                    .as_deref()
                    .expect("keyed response was bounded"),
            )
            .await
            .context("Failed to commit durable text TTS idempotency result")?;
            if !committed {
                tx.rollback().await?;
                return Ok(DurableTextTtsAcceptanceOutcome::ReservationLost);
            }
            self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Idempotency)?;
        }

        let job = get_job_with(&tx, &job_id)
            .await?
            .ok_or_else(|| anyhow!("Created durable text TTS job was not found"))?;
        let stage = get_stage_with(&tx, &stage_id)
            .await?
            .ok_or_else(|| anyhow!("Created durable text TTS stage was not found"))?;
        let input_artifact = get_artifact_with(&tx, &artifact_id)
            .await?
            .ok_or_else(|| anyhow!("Created durable text TTS artifact was not found"))?;
        tx.commit()
            .await
            .context("Failed to commit durable text TTS acceptance transaction")?;
        self.wake_claimers();

        Ok(DurableTextTtsAcceptanceOutcome::Committed(
            DurableTextTtsAcceptance {
                record,
                job,
                stage,
                input_artifact,
            },
        ))
    }

    /// Atomically accept one queued job graph: its text input, the job, its
    /// input artifacts, every stage and, when keyed, the idempotency result.
    /// No worker can claim a stage, and no job can exist without its stages,
    /// before the whole graph commits. Speech jobs take admission inside the
    /// same transaction.
    pub async fn accept_durable_job(
        &self,
        input: NewDurableJobAcceptance,
    ) -> anyhow::Result<DurableJobAcceptanceOutcome> {
        let NewDurableJobAcceptance {
            mut job,
            text_input,
            artifacts,
            stages,
            reservation,
            idempotency_response,
            idempotency_retention_ms,
        } = input;
        anyhow::ensure!(
            job.status == RuntimeJobStatus::Queued,
            StoreError::invalid("Durable job acceptance requires a queued job")
        );
        anyhow::ensure!(
            !stages.is_empty(),
            StoreError::invalid("Durable job acceptance requires at least one stage")
        );
        anyhow::ensure!(
            job.max_attempts > 0 && stages.iter().all(|stage| stage.max_attempts > 0),
            StoreError::invalid("Durable job acceptance requires at least one attempt")
        );
        let text_input_id = text_input.as_ref().map(|text| text.id.clone());
        anyhow::ensure!(
            job.input_text_asset_id == text_input_id,
            StoreError::invalid("Durable job text input must be the job's text input")
        );
        let mut artifact_ids = std::collections::HashSet::new();
        for artifact in &artifacts {
            validate_bounded_field("artifact ID", &artifact.id, 64)?;
            anyhow::ensure!(
                artifact_ids.insert(artifact.id.as_str()),
                StoreError::invalid("Durable job artifact IDs must be unique")
            );
            anyhow::ensure!(
                artifact.text_asset_id.is_none() || artifact.text_asset_id == text_input_id,
                StoreError::invalid(
                    "Durable job artifacts may only reference the job's text input"
                )
            );
        }
        let mut sequences = std::collections::HashSet::new();
        for stage in &stages {
            validate_bounded_field("stage kind", &stage.stage_kind, 128)?;
            anyhow::ensure!(
                sequences.insert(stage.sequence),
                StoreError::invalid("Durable job stage sequences must be unique")
            );
            anyhow::ensure!(
                stage
                    .input_artifact_ids
                    .iter()
                    .all(|id| artifact_ids.contains(id.as_str())),
                StoreError::invalid("Durable job stages may only consume the job's artifacts")
            );
        }
        let response_json = match reservation.as_ref() {
            Some(reservation) => {
                validate_acceptance_reservation(reservation, idempotency_retention_ms)?;
                anyhow::ensure!(
                    job.idempotency_key.is_none()
                        || job.idempotency_key.as_ref() == Some(&reservation.idempotency_key),
                    StoreError::invalid("Durable job idempotency key must match its reservation")
                );
                job.idempotency_key = Some(reservation.idempotency_key.clone());
                Some(bounded_json_string(
                    &idempotency_response,
                    MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
                    "Durable idempotency result",
                )?)
            }
            None => None,
        };
        let admission_tenant = if job.job_kind == RuntimeJobKind::TtsSpeech {
            Some(speech_admission_tenant(&job.request_json)?)
        } else {
            None
        };
        let request_json = json_to_db_string(&job.request_json, "{}")?;
        let model_snapshot_json = json_to_db_string(&job.model_snapshot_json, "{}")?;
        let retry_policy_json = json_to_db_string(&job.retry_policy_json, "{}")?;

        let now = self.now_millis();
        let now_u64 = nonnegative_timestamp(now)?;
        let job_id = new_uuid();
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start durable job acceptance transaction")?;

        if let Some(reservation) = reservation.as_ref() {
            if !durable_reservation_is_live_with(&tx, reservation, now_u64).await? {
                tx.rollback().await?;
                return Ok(DurableJobAcceptanceOutcome::ReservationLost);
            }
        }
        if let Some(tenant) = admission_tenant.as_deref() {
            self.check_tts_admission(&tx, tenant).await?;
        }

        if let Some(text) = text_input {
            let normalized_text = text
                .asset
                .normalized_text
                .unwrap_or_else(|| text.asset.raw_text.clone());
            let character_count = normalized_text.chars().count() as u64;
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                INSERT INTO text_assets (
                    id, created_at, updated_at, raw_text, normalized_text,
                    language_hint, character_count, sha256, safety_status,
                    retention_policy, structure_json
                )
                VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
                vec![
                    text.id.into(),
                    now.into(),
                    text.asset.raw_text.into(),
                    normalized_text.into(),
                    opt_string(text.asset.language_hint),
                    u64_to_i64_value(character_count)?,
                    opt_string(text.asset.sha256),
                    text.asset.safety_status.into(),
                    text.asset.retention_policy.into(),
                    json_to_db_string(&text.asset.structure_json, "{}")?.into(),
                ],
            )?)
            .await
            .context("Failed to create durable job text input")?;
            self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::TextAsset)?;
        }

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_jobs (
                id, created_at, updated_at, queued_at, started_at, finished_at,
                job_kind, status, priority, model_id, capability,
                route_record_kind, route_record_id, input_media_asset_id,
                input_text_asset_id, request_json, model_snapshot_json,
                progress_json, error_code, error_message, attempt_count,
                max_attempts, retry_policy_json, idempotency_key,
                correlation_id, cancellation_reason, admission_tenant
            )
            VALUES (?1, ?2, ?2, ?2, NULL, NULL, ?3, 'queued', ?4, ?5, ?6, ?7,
                    ?8, ?9, ?10, ?11, ?12, NULL, NULL, NULL, 0, ?13, ?14, ?15,
                    ?16, NULL, ?17)
            "#,
            vec![
                job_id.clone().into(),
                now.into(),
                job.job_kind.as_db_value().into(),
                job.priority.into(),
                opt_string(job.model_id),
                opt_string(job.capability),
                opt_string(job.route_record_kind),
                opt_string(job.route_record_id),
                opt_string(job.input_media_asset_id),
                opt_string(job.input_text_asset_id),
                request_json.into(),
                model_snapshot_json.into(),
                u32_to_i64_value(job.max_attempts).into(),
                retry_policy_json.into(),
                opt_string(job.idempotency_key),
                opt_string(job.correlation_id),
                opt_string(admission_tenant),
            ],
        )?)
        .await
        .context("Failed to create durable runtime job")?;
        self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Job)?;

        let artifact_ids = artifacts
            .iter()
            .map(|artifact| artifact.id.clone())
            .collect::<Vec<_>>();
        for artifact in artifacts {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                INSERT INTO runtime_artifacts (
                    id, job_id, stage_id, created_at, artifact_kind,
                    artifact_role, media_asset_id, text_asset_id, storage_key,
                    content_type, filename, size_bytes, sha256, metadata_json,
                    retention_policy
                )
                VALUES (?1, ?2, NULL, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                        ?13, ?14)
                "#,
                vec![
                    artifact.id.into(),
                    job_id.clone().into(),
                    now.into(),
                    artifact.artifact_kind.as_db_value().into(),
                    artifact.artifact_role.as_db_value().into(),
                    opt_string(artifact.media_asset_id),
                    opt_string(artifact.text_asset_id),
                    opt_string(artifact.storage_key),
                    opt_string(artifact.content_type),
                    opt_string(artifact.filename),
                    opt_u64(artifact.size_bytes),
                    opt_string(artifact.sha256),
                    json_to_db_string(&artifact.metadata_json, "{}")?.into(),
                    artifact.retention_policy.into(),
                ],
            )?)
            .await
            .context("Failed to create durable job input artifact")?;
            self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Artifact)?;
        }

        let mut stage_ids = Vec::with_capacity(stages.len());
        for stage in stages {
            let stage_id = new_uuid();
            let resource_hints = stage.resource_hints.normalized();
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                INSERT INTO job_stages (
                    id, job_id, created_at, updated_at, sequence, stage_kind,
                    queue_class, resource_hints_json, resource_target,
                    required_backend, required_device_class,
                    min_resource_memory_bytes, resource_concurrency_weight,
                    resource_hints_version,
                    status, capability, model_id, worker_id, lease_expires_at,
                    available_at, attempt_token, attempt_count, max_attempts,
                    input_artifact_ids_json, output_artifact_ids_json,
                    progress_json, started_at, finished_at, error_code,
                    error_message, cancellation_state
                )
                VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                        ?17, 'queued', ?13, ?14, NULL, NULL, ?3, NULL, 0, ?15, ?16,
                        '[]', NULL, NULL, NULL, NULL, NULL, NULL)
                "#,
                vec![
                    stage_id.clone().into(),
                    job_id.clone().into(),
                    now.into(),
                    u32_to_i64_value(stage.sequence).into(),
                    stage.stage_kind.into(),
                    stage.queue_class.as_db_value().into(),
                    json_to_db_string(&json!(resource_hints), "{}")?.into(),
                    resource_hints.target.as_db_value().into(),
                    opt_string(
                        resource_hints
                            .backend
                            .map(|backend| backend.as_db_value().to_string()),
                    ),
                    opt_string(
                        resource_hints
                            .device_class
                            .map(|device| device.as_db_value().to_string()),
                    ),
                    opt_u64(resource_hints.min_memory_bytes),
                    u32_to_i64_value(resource_hints.concurrency_weight).into(),
                    opt_string(stage.capability),
                    opt_string(stage.model_id),
                    u32_to_i64_value(stage.max_attempts).into(),
                    json_to_db_string(&json!(stage.input_artifact_ids), "[]")?.into(),
                    i64::from(resource_hints.version).into(),
                ],
            )?)
            .await
            .context("Failed to create durable job stage")?;
            stage_ids.push(stage_id);
            self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Stage)?;
        }

        if let (Some(reservation), Some(response_json)) =
            (reservation.as_ref(), response_json.as_deref())
        {
            let committed = commit_durable_reservation_with(
                &tx,
                reservation,
                now_u64,
                idempotency_retention_ms,
                &job_id,
                response_json,
            )
            .await
            .context("Failed to commit durable job idempotency result")?;
            if !committed {
                tx.rollback().await?;
                return Ok(DurableJobAcceptanceOutcome::ReservationLost);
            }
            self.inject_durable_acceptance_failure(DurableAcceptanceFailpoint::Idempotency)?;
        }

        let job = get_job_with(&tx, &job_id)
            .await?
            .ok_or_else(|| anyhow!("Accepted durable job was not found"))?;
        let mut artifacts = Vec::with_capacity(artifact_ids.len());
        for id in &artifact_ids {
            artifacts.push(
                get_artifact_with(&tx, id)
                    .await?
                    .ok_or_else(|| anyhow!("Accepted durable job artifact was not found"))?,
            );
        }
        let mut stages = Vec::with_capacity(stage_ids.len());
        for id in &stage_ids {
            stages.push(
                get_stage_with(&tx, id)
                    .await?
                    .ok_or_else(|| anyhow!("Accepted durable job stage was not found"))?,
            );
        }
        tx.commit()
            .await
            .context("Failed to commit durable job acceptance transaction")?;
        self.wake_claimers();

        Ok(DurableJobAcceptanceOutcome::Committed(
            DurableJobAcceptance {
                job,
                artifacts,
                stages,
            },
        ))
    }
}

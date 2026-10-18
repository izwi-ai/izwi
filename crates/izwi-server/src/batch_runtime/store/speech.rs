//! Speech history settlement and deletion, TTS admission and the PCM replay journal.

use super::*;

/// Scalar completion data for the durable Fish final-audio settlement.
///
/// The provider object is written before this metadata transaction starts.
/// This value carries only bounded route projection fields and the final
/// checkpoint marker; it deliberately does not contain audio bytes.
#[derive(Debug, Clone)]
pub(crate) struct SpeechHistoryFinalization {
    pub route_kind: SpeechRouteKind,
    pub record_id: String,
    pub model_id: Option<String>,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub input_text: String,
    pub voice_description: Option<String>,
    pub reference_text: Option<String>,
    pub generation_time_ms: f64,
    pub audio_duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub tokens_generated: Option<usize>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub artifact_metadata_json: serde_json::Value,
    pub progress: serde_json::Value,
}

pub(super) fn speech_admission_tenant(request: &serde_json::Value) -> anyhow::Result<String> {
    match request.get("tenant_key").filter(|value| !value.is_null()) {
        Some(value) => {
            let key: [u8; 32] = serde_json::from_value(value.clone())
                .context("Invalid server-authored speech tenant identity")?;
            Ok(key.iter().map(|byte| format!("{byte:02x}")).collect())
        }
        None => Ok("anonymous".into()),
    }
}

impl BatchRuntimeStore {
    /// Atomically remove one speech-history row while tombstoning its exact
    /// tenant-scoped opaque audio and retaining the physical cleanup intent.
    pub(crate) async fn delete_opaque_speech_history_record(
        &self,
        route_kind: &str,
        record_id: &str,
        media_asset_id: &str,
        tenant_scope: &str,
    ) -> anyhow::Result<Option<ArtifactCleanupIntent>> {
        validate_artifact_cleanup_tenant(tenant_scope)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start opaque speech-history deletion transaction")?;
        lock_named(&tx, TTS_ADMISSION_LOCK).await?;
        let lock_clause = match tx.get_database_backend() {
            DbBackend::Sqlite => "",
            DbBackend::Postgres => " FOR UPDATE OF h, m",
            backend => bail!("Unsupported speech-history database backend: {backend:?}"),
        };
        let relation_sql = format!(
            r#"
            SELECT m.id, m.metadata_json, m.retention_policy, m.content_type,
                   m.filename, h.audio_mime_type, h.audio_filename
            FROM speech_history_records h
            JOIN media_assets m ON m.id = h.audio_media_asset_id
            WHERE h.route_kind = ?1
              AND h.id = ?2
              AND h.audio_storage_path = ''
              AND h.audio_media_asset_id = ?3
              AND h.audio_artifact_tenant = ?4
              AND h.processing_status = 'ready'
              AND m.asset_kind = 'opaque_artifact'
              AND m.storage_namespace = 'artifact_store_v1'
              AND m.deleted_at IS NULL
            LIMIT 1{lock_clause}
            "#
        );
        let relation = tx
            .query_one_raw(raw::statement(
                &tx,
                relation_sql,
                vec![
                    route_kind.into(),
                    record_id.into(),
                    media_asset_id.into(),
                    tenant_scope.into(),
                ],
            )?)
            .await
            .context("Failed to lock opaque speech-history audio relation")?;
        let Some(relation) = relation else {
            tx.rollback().await?;
            return Ok(None);
        };
        let metadata_json: String = relation.try_get_by_index(1)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
            .context("Opaque artifact tenant metadata is invalid")?;
        anyhow::ensure!(
            metadata
                .get("artifact_store")
                .and_then(|value| value.get("version"))
                .and_then(serde_json::Value::as_u64)
                == Some(1)
                && metadata
                    .get("artifact_store")
                    .and_then(|value| value.get("tenant_id"))
                    .and_then(serde_json::Value::as_str)
                    == Some(tenant_scope),
            "Opaque speech-history artifact is not owned by the requested tenant"
        );
        let retention_policy: String = relation.try_get_by_index(2)?;
        let media_content_type: String = relation.try_get_by_index(3)?;
        let media_filename: Option<String> = relation.try_get_by_index(4)?;
        let history_content_type: String = relation.try_get_by_index(5)?;
        let history_filename: Option<String> = relation.try_get_by_index(6)?;
        anyhow::ensure!(
            retention_policy == "artifact_durable"
                && media_content_type == history_content_type
                && media_filename == history_filename,
            "Opaque speech-history artifact metadata does not match a durable record"
        );

        let wrong_tenant_job = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT 1
                FROM runtime_jobs
                WHERE job_kind = 'tts_speech'
                  AND route_record_kind = ?1
                  AND route_record_id = ?2
                  AND COALESCE(admission_tenant, 'anonymous') <> ?3
                LIMIT 1
                "#,
                vec![route_kind.into(), record_id.into(), tenant_scope.into()],
            )?)
            .await
            .context("Failed to validate speech-history runtime tenant ownership")?;
        anyhow::ensure!(
            wrong_tenant_job.is_none(),
            "Speech-history runtime ownership does not match the artifact tenant"
        );
        let active_job = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT 1
                FROM runtime_jobs
                WHERE job_kind = 'tts_speech'
                  AND route_record_kind = ?1
                  AND route_record_id = ?2
                  AND COALESCE(admission_tenant, 'anonymous') = ?3
                  AND status IN ('queued', 'running', 'retrying')
                LIMIT 1
                "#,
                vec![route_kind.into(), record_id.into(), tenant_scope.into()],
            )?)
            .await
            .context("Failed to inspect active speech-history runtime ownership")?;
        anyhow::ensure!(
            active_job.is_none(),
            "Opaque speech-history record is still owned by an active runtime job"
        );
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE runtime_jobs
            SET error_code = 'speech_replay_expired'
            WHERE job_kind = 'tts_speech'
              AND route_record_kind = ?1
              AND route_record_id = ?2
              AND COALESCE(admission_tenant, 'anonymous') = ?3
              AND status IN ('failed', 'cancelled')
            "#,
            vec![route_kind.into(), record_id.into(), tenant_scope.into()],
        )?)
        .await
        .context("Failed to fence speech-history runtime retries")?;

        // Remove the protected reference first inside this transaction. The
        // tombstone helper rejects every still-referenced asset; any later
        // error rolls this deletion back with the tombstone and outbox write.
        let deleted = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                DELETE FROM speech_history_records
                WHERE route_kind = ?1
                  AND id = ?2
                  AND audio_storage_path = ''
                  AND audio_media_asset_id = ?3
                  AND audio_artifact_tenant = ?4
                "#,
                vec![
                    route_kind.into(),
                    record_id.into(),
                    media_asset_id.into(),
                    tenant_scope.into(),
                ],
            )?)
            .await
            .context("Failed to delete opaque speech-history row")?;
        if deleted.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        let Some((_, intent)) = tombstone_media_asset_with_cleanup_in(
            &tx,
            media_asset_id,
            tenant_scope,
            ArtifactCleanupReason::ArtifactDeleted,
            self.artifact_cleanup_limits(),
            self.now_millis(),
        )
        .await?
        else {
            tx.rollback().await?;
            return Ok(None);
        };
        tx.commit()
            .await
            .context("Failed to commit opaque speech-history deletion")?;
        Ok(Some(intent))
    }

    /// Atomically settle a durable Fish speech attempt's final WAV.
    ///
    /// The provider write has already been verified and recorded as `stored`
    /// before this method is called. This transaction consumes that exact
    /// reservation, creates the opaque media asset and attempt output, marks
    /// the route projection ready, and completes the stage/job together. A
    /// lost attempt leaves the provider reservation for the bounded cleanup
    /// worker; it never deletes a provider object while execution could still
    /// own it.
    pub(crate) async fn settle_reserved_speech_history_audio(
        &self,
        reservation: &ProviderWriteReservation,
        lease: &StageLease,
        media: NewMediaAsset,
        mut artifact: NewStageOutputArtifact,
        finalization: SpeechHistoryFinalization,
    ) -> anyhow::Result<Option<ReservedStageArtifactPublication>> {
        const FINAL_PUBLICATION_KEY: &str = "speech-final-audio/v1";

        anyhow::ensure!(
            media.storage_key.as_str() == reservation.storage_key.as_deref().unwrap_or_default()
                && media.content_type == reservation.content_type
                && media.filename == reservation.filename
                && media.size_bytes == reservation.expected_size_bytes
                && media.sha256.as_deref() == Some(reservation.expected_sha256.as_str()),
            "Provider write publication did not match its reservation"
        );
        anyhow::ensure!(
            artifact.publication_key == FINAL_PUBLICATION_KEY
                && artifact.media_asset_id.is_none()
                && artifact.text_asset_id.is_none()
                && artifact.storage_key.is_none()
                && artifact.artifact_kind == RuntimeArtifactKind::Audio
                && artifact.artifact_role == RuntimeArtifactRole::OutputPrimary
                && artifact.retention_policy == "speech_history",
            "Speech final artifact has an invalid publication contract"
        );
        let Some(attempt_token) = lease.attempt_token.as_deref() else {
            return Ok(None);
        };
        anyhow::ensure!(
            finalization.record_id.len() <= 128 && !finalization.record_id.trim().is_empty(),
            "Speech finalization record id is invalid"
        );
        anyhow::ensure!(
            finalization.progress.is_object()
                && finalization
                    .progress
                    .get("final_audio_published")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true),
            "Speech finalization progress lacks the final-audio marker"
        );
        let artifact_metadata_bytes = serde_json::to_vec(&artifact.metadata_json)
            .map_err(|_| anyhow!("Speech final artifact metadata is invalid"))?;
        anyhow::ensure!(
            artifact_metadata_bytes.len() <= 8 * 1024,
            "Speech final artifact metadata exceeds 8 KiB"
        );
        let artifact_metadata_bytes = serde_json::to_vec(&finalization.artifact_metadata_json)
            .map_err(|_| anyhow!("Invalid speech final artifact metadata"))?;
        anyhow::ensure!(
            artifact_metadata_bytes.len() <= 8 * 1024,
            "Speech final artifact metadata exceeds 8 KiB"
        );
        let progress_json = bounded_json_string(
            &finalization.progress,
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Stage progress",
        )?;
        let finalization_input_text = finalization.input_text.trim().to_string();
        anyhow::ensure!(
            finalization_input_text.len() <= MAX_DURABLE_TTS_TEXT_BYTES,
            "Speech finalization input text exceeds the durable limit"
        );
        let model_id = sanitize_optional_text(finalization.model_id.as_deref(), 160);
        let speaker = sanitize_optional_text(finalization.speaker.as_deref(), 120);
        let language = sanitize_optional_text(finalization.language.as_deref(), 80);
        let saved_voice_id = sanitize_optional_text(finalization.saved_voice_id.as_deref(), 160);
        let speed = finalization
            .speed
            .filter(|value| value.is_finite() && *value > 0.0);
        let voice_description =
            sanitize_optional_text(finalization.voice_description.as_deref(), 2_000);
        let reference_text = sanitize_optional_text(finalization.reference_text.as_deref(), 2_000);
        let generation_time_ms = if finalization.generation_time_ms.is_finite() {
            finalization.generation_time_ms.max(0.0)
        } else {
            0.0
        };
        let audio_duration_secs = finalization
            .audio_duration_secs
            .filter(|value| value.is_finite() && *value >= 0.0);
        let rtf = finalization
            .rtf
            .filter(|value| value.is_finite() && *value >= 0.0);
        let tokens_generated = finalization
            .tokens_generated
            .filter(|value| *value > 0)
            .and_then(|value| i64::try_from(value).ok());
        let audio_mime_type = sanitize_audio_mime_type(finalization.audio_mime_type.as_str());
        let audio_filename = sanitize_optional_text(finalization.audio_filename.as_deref(), 260);
        anyhow::ensure!(
            audio_mime_type == media.content_type
                && audio_filename == media.filename
                && media.retention_policy == "artifact_durable",
            "Speech final audio metadata does not match its durable artifact"
        );
        artifact.content_type = Some(media.content_type.clone());
        artifact.filename = media.filename.clone();
        artifact.size_bytes = Some(media.size_bytes);
        artifact.sha256 = media.sha256.clone();
        let expected_media = media.clone();
        let expected_artifact = artifact.clone();
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start speech final audio settlement transaction")?;
        lock_stage_job_with(&tx, &lease.stage_id).await?;
        let now = self.now_millis();
        let row = get_provider_write_with(&tx, &reservation.write_id).await?;
        if row.as_ref() != Some(reservation) {
            tx.rollback().await?;
            return Ok(None);
        }
        let lock_clause = match tx.get_database_backend() {
            DbBackend::Sqlite => "",
            DbBackend::Postgres => " FOR UPDATE OF s, j, h",
            backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
        };
        let owns_stage_sql = format!(
            r#"
            SELECT s.id
            FROM job_stages s
            JOIN runtime_jobs j ON j.id = s.job_id
            JOIN speech_history_records h
              ON h.id = j.route_record_id
             AND h.route_kind = j.route_record_kind
            WHERE s.id = ?1
              AND s.status = 'running'
              AND s.cancellation_state IS NULL
              AND s.worker_id = ?2
              AND s.attempt_count = ?3
              AND s.attempt_token = ?4
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?5
              AND j.job_kind = 'tts_speech'
              AND j.route_record_kind = ?6
              AND j.route_record_id = ?7
              AND j.status IN ('queued', 'running', 'retrying')
              AND j.cancellation_state IS NULL
              AND COALESCE(j.admission_tenant, 'anonymous') = ?8
              AND h.id = ?7
              AND h.route_kind = ?6
              AND h.processing_status IN ('pending', 'processing')
              AND h.runtime_stage_id = s.id
              AND h.runtime_attempt_token = s.attempt_token
              AND h.audio_storage_path = ''
              AND h.audio_media_asset_id IS NULL
              AND h.audio_artifact_tenant IS NULL
            LIMIT 1{lock_clause}
            "#
        );
        let owns_stage = tx
            .query_one_raw(raw::statement(
                &tx,
                owns_stage_sql,
                vec![
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.into(),
                    now.into(),
                    finalization.route_kind.as_db_value().into(),
                    finalization.record_id.clone().into(),
                    reservation.tenant_scope.clone().into(),
                ],
            )?)
            .await
            .context("Failed to lock speech final audio ownership")?
            .is_some();
        if !owns_stage {
            if !mark_provider_write_cleanup_pending_with(
                &tx,
                reservation,
                now,
                "Speech final audio attempt lost ownership before settlement",
            )
            .await?
            {
                tx.rollback().await?;
                return Ok(None);
            }
            tx.commit().await?;
            return Ok(None);
        }

        let existing =
            stage_output_for_key_with(&tx, lease, attempt_token, FINAL_PUBLICATION_KEY).await?;
        let (media_id, published) = if let Some(existing) = existing {
            let media_id = existing
                .media_asset_id
                .clone()
                .context("Speech final artifact is missing its media asset")?;
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
            anyhow::ensure!(
                exact,
                "Speech final publication key was reused for a different opaque artifact"
            );
            (media_id, existing)
        } else {
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
                    media.asset_kind.clone().into(),
                    media.storage_namespace.clone().into(),
                    media.storage_key.clone().into(),
                    media.content_type.clone().into(),
                    opt_string(media.filename.clone()),
                    u64_to_i64_value(media.size_bytes)?,
                    opt_string(media.sha256.clone()),
                    opt_f64(media.duration_secs),
                    opt_u32(media.sample_rate_hz),
                    opt_u16(media.channel_count),
                    opt_f32(media.peak_amplitude),
                    opt_f32(media.rms_amplitude),
                    opt_string(media.source_asset_id.clone()),
                    opt_string(media.canonical_profile_version.clone()),
                    media.scan_status.clone().into(),
                    media.retention_policy.clone().into(),
                    media_metadata_json.into(),
                ],
            )?)
            .await
            .context("Failed to create speech final opaque media asset")?;

            artifact.media_asset_id = Some(media_id.clone());
            let artifact_id = new_uuid();
            let artifact_metadata_json = json_to_db_string(&artifact.metadata_json, "{}")?;
            tx.execute_raw(raw::statement(
                &tx,
                r#"
            INSERT INTO runtime_artifacts (
                id, job_id, stage_id, producer_attempt_count,
                producer_attempt_token, publication_key, created_at,
                artifact_kind, artifact_role, media_asset_id, text_asset_id,
                storage_key, content_type, filename, size_bytes, sha256,
                metadata_json, retention_policy
            )
            SELECT ?1, s.job_id, s.id, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                   NULL, NULL, ?9, ?10, ?11, ?12, ?13, ?14
            FROM job_stages s
            JOIN runtime_jobs j ON j.id = s.job_id
            WHERE s.id = ?15
              AND s.status = 'running'
              AND s.cancellation_state IS NULL
              AND s.worker_id = ?16
              AND s.attempt_count = ?2
              AND s.attempt_token = ?3
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?5
              AND j.job_kind = 'tts_speech'
              AND j.route_record_kind = ?17
              AND j.route_record_id = ?18
              AND j.status IN ('queued', 'running', 'retrying')
              AND j.cancellation_state IS NULL
              AND COALESCE(j.admission_tenant, 'anonymous') = ?19
                "#,
                vec![
                    artifact_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.into(),
                    FINAL_PUBLICATION_KEY.into(),
                    now.into(),
                    artifact.artifact_kind.as_db_value().into(),
                    artifact.artifact_role.as_db_value().into(),
                    media_id.clone().into(),
                    opt_string(artifact.content_type.clone()),
                    opt_string(artifact.filename.clone()),
                    opt_u64(artifact.size_bytes),
                    opt_string(artifact.sha256.clone()),
                    artifact_metadata_json.into(),
                    artifact.retention_policy.clone().into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    finalization.route_kind.as_db_value().into(),
                    finalization.record_id.clone().into(),
                    reservation.tenant_scope.clone().into(),
                ],
            )?)
            .await
            .context("Failed to attach speech final opaque artifact")?;
            let Some(published) =
                stage_output_for_key_with(&tx, lease, attempt_token, FINAL_PUBLICATION_KEY).await?
            else {
                tx.rollback().await?;
                return Ok(None);
            };
            anyhow::ensure!(
                published.media_asset_id.as_deref() == Some(media_id.as_str()),
                "Speech final artifact publication was superseded"
            );
            (media_id, published)
        };

        let history_result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE speech_history_records
                SET processing_status = 'ready',
                    processing_error = NULL,
                    model_id = ?1,
                    speaker = ?2,
                    language = ?3,
                    saved_voice_id = ?4,
                    speed = ?5,
                    input_text = ?6,
                    voice_description = ?7,
                    reference_text = ?8,
                    generation_time_ms = ?9,
                    audio_duration_secs = ?10,
                    rtf = ?11,
                    tokens_generated = ?12,
                    audio_mime_type = ?13,
                    audio_filename = ?14,
                    audio_storage_path = '',
                    audio_media_asset_id = ?15,
                    audio_artifact_tenant = ?16
                WHERE route_kind = ?17
                  AND id = ?18
                  AND processing_status IN ('pending', 'processing')
                  AND runtime_stage_id = ?19
                  AND runtime_attempt_token = ?20
                  AND audio_storage_path = ''
                  AND audio_media_asset_id IS NULL
                  AND audio_artifact_tenant IS NULL
                "#,
                vec![
                    opt_string(model_id),
                    opt_string(speaker),
                    opt_string(language),
                    opt_string(saved_voice_id),
                    opt_f64(speed),
                    finalization_input_text.into(),
                    opt_string(voice_description),
                    opt_string(reference_text),
                    generation_time_ms.into(),
                    opt_f64(audio_duration_secs),
                    opt_f64(rtf),
                    opt_i64(tokens_generated),
                    audio_mime_type.into(),
                    opt_string(audio_filename),
                    media_id.clone().into(),
                    reservation.tenant_scope.clone().into(),
                    finalization.route_kind.as_db_value().into(),
                    finalization.record_id.clone().into(),
                    lease.stage_id.clone().into(),
                    attempt_token.into(),
                ],
            )?)
            .await
            .context("Failed to settle speech history route projection")?;
        if history_result.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }

        let output_json = json_to_db_string(&json!([published.id.clone()]), "[]")?;
        let stage_result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET status = 'completed',
                    cancellation_state = NULL,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    output_artifact_ids_json = ?2,
                    progress_json = ?3,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?4
                  AND status = 'running'
                  AND cancellation_state IS NULL
                  AND attempt_count = ?5
                  AND attempt_token = ?6
                  AND worker_id = ?7
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?1
                "#,
                vec![
                    now.into(),
                    output_json.into(),
                    progress_json.into(),
                    lease.stage_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.into(),
                    lease.worker_id.clone().into(),
                ],
            )?)
            .await
            .context("Failed to complete speech final stage")?;
        if stage_result.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        complete_job_if_all_stages_finished_with(&tx, &published.job_id, now).await?;
        let deleted = tx
            .execute_raw(raw::statement(
                &tx,
                "DELETE FROM provider_write_reservations WHERE write_id = ?1 AND reservation_token = ?2 AND state = 'stored'",
                vec![
                    reservation.write_id.clone().into(),
                    reservation.reservation_token.clone().into(),
                ],
            )?)
            .await?;
        if deleted.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        release_quota_with(&tx, QuotaKind::ProviderWrite, &reservation.tenant_scope, 1).await?;
        let asset = get_media_asset_with(&tx, &media_id)
            .await?
            .context("Settled speech final media asset was not found")?;
        tx.commit()
            .await
            .context("Failed to commit speech final audio settlement")?;
        Ok(Some(ReservedStageArtifactPublication::Published {
            asset: Box::new(asset),
            artifact: published,
        }))
    }

    pub(super) fn tts_admission_limits(&self) -> (usize, usize) {
        #[cfg(test)]
        if let Some(limits) = *self
            .test_tts_admission_limits
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
        {
            return limits;
        }
        let tuning = crate::env_config::tuning();
        (
            tuning.tts_max_active_jobs,
            tuning.tts_max_active_jobs_per_tenant,
        )
    }

    pub(super) async fn check_tts_admission(
        &self,
        tx: &sea_orm::DatabaseTransaction,
        tenant: &str,
    ) -> anyhow::Result<()> {
        // The unique insert and row update serialize admission across processes and
        // PostgreSQL connections; a read/count alone would allow concurrent overflow.
        lock_named(tx, TTS_ADMISSION_LOCK).await?;
        let row = tx.query_one_raw(raw::statement(tx,
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN COALESCE(admission_tenant, 'anonymous') = ?1 THEN 1 ELSE 0 END), 0) FROM runtime_jobs WHERE job_kind = 'tts_speech' AND status IN ('queued','running','retrying')",
            vec![tenant.into()],
        )?).await?.ok_or_else(|| anyhow!("Speech admission count returned no row"))?;
        let active = u64::try_from(row.try_get_by_index::<i64>(0)?)?;
        let owned = u64::try_from(row.try_get_by_index::<i64>(1)?)?;
        let (global_limit, tenant_limit) = self.tts_admission_limits();
        if active >= global_limit as u64 || owned >= tenant_limit as u64 {
            return Err(StoreError::capacity(format!("Speech job admission capacity exhausted: active={active}/{global_limit}, tenant={owned}/{tenant_limit}")).into());
        }
        Ok(())
    }

    /// Avoid expensive input ingestion during overload. The create transaction
    /// still performs the authoritative check because preflight reserves no slot.
    pub async fn preflight_speech_admission(
        &self,
        tenant_key: Option<[u8; 32]>,
    ) -> anyhow::Result<()> {
        let tenant = speech_admission_tenant(&json!({"tenant_key": tenant_key}))?;
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        self.check_tts_admission(&tx, &tenant).await?;
        tx.rollback().await?;
        Ok(())
    }

    /// Claim expired replay for deletion before exposing storage keys to GC.
    /// A terminal job loses retry eligibility atomically with this claim. Retrying
    /// and GC both conditionally update the job row, fencing their race on every DB.
    ///
    /// Terminal jobs are visited oldest first through the replay-sweep
    /// index; a job whose journal is empty is marked swept and never scanned
    /// again, so each pass reads a bounded page instead of every artifact.
    pub async fn expired_speech_pcm(&self, before: u64) -> anyhow::Result<Vec<RuntimeArtifact>> {
        const MAX_JOBS: i64 = 16;
        const MAX_ARTIFACTS: usize = 64;
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let cutoff = i64::try_from(before)?;
        let now = self.now_millis();
        let jobs = tx
            .query_all_raw(raw::statement(
                &tx,
                r#"
                SELECT id FROM runtime_jobs
                WHERE replay_swept_at IS NULL AND updated_at < ?1
                  AND status IN ('completed','failed','cancelled')
                ORDER BY updated_at ASC, id ASC
                LIMIT ?2
                "#,
                vec![cutoff.into(), MAX_JOBS.into()],
            )?)
            .await?;
        // A byte range over `speech-pcm/…` keys: publication_key is
        // COLLATE "C" on PostgreSQL (app migration 6), BINARY on SQLite.
        let page_sql = concat!(
            select_runtime_artifacts!(),
            " WHERE job_id = ?1 AND publication_key >= 'speech-pcm/'",
            " AND publication_key < 'speech-pcm0' ORDER BY publication_key ASC LIMIT ?2"
        );
        let mut artifacts = Vec::new();
        for row in jobs {
            let remaining = MAX_ARTIFACTS - artifacts.len();
            if remaining == 0 {
                break;
            }
            let job_id: String = row.try_get_by_index(0)?;
            let page = tx
                .query_all_raw(raw::statement(
                    &tx,
                    page_sql,
                    vec![job_id.clone().into(), i64::try_from(remaining)?.into()],
                )?)
                .await?
                .iter()
                .map(map_runtime_artifact)
                .collect::<anyhow::Result<Vec<_>>>()?;
            if page.is_empty() {
                tx.execute_raw(raw::statement(
                    &tx,
                    "UPDATE runtime_jobs SET replay_swept_at = ?1 WHERE id = ?2 AND status IN ('completed','failed','cancelled')",
                    vec![now.into(), job_id.into()],
                )?)
                .await?;
                continue;
            }
            let claimed = tx.execute_raw(raw::statement(&tx,
                "UPDATE runtime_jobs SET error_code = 'speech_replay_expired' WHERE id = ?1 AND status IN ('completed','failed','cancelled') AND updated_at < ?2",
                vec![job_id.into(), cutoff.into()],
            )?).await?;
            if claimed.rows_affected() == 1 {
                artifacts.extend(page);
            }
        }
        tx.commit().await?;
        Ok(artifacts)
    }

    /// Permanently fence manual replay deletion against concurrent job retry.
    /// The retry transition tests this marker in its conditional UPDATE too.
    pub async fn fence_speech_replay_deletion(&self, job_id: &str) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let result = db.execute_raw(raw::statement(db,
            "UPDATE runtime_jobs SET error_code = 'speech_replay_expired' WHERE id = ?1 AND status IN ('completed','failed','cancelled')",
            vec![job_id.into()],
        )?).await?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn remove_speech_pcm_artifact(&self, id: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "DELETE FROM runtime_artifacts WHERE id = ?1 AND publication_key LIKE 'speech-pcm/%'",
            vec![id.into()],
        )?)
        .await?;
        Ok(())
    }

    /// Read a bounded page of committed speech PCM without loading the whole journal.
    pub async fn speech_pcm_after(
        &self,
        job_id: &str,
        after_sequence: Option<u64>,
        limit: u32,
    ) -> anyhow::Result<Vec<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let after_key = after_sequence
            .map(crate::batch_runtime::speech_progress::pcm_publication_key)
            .unwrap_or_else(|| "speech-pcm/".to_string());
        let sql = concat!(
            select_runtime_artifacts!(),
            " WHERE job_id = ?1 AND publication_key > ?2",
            " AND publication_key < 'speech-pcm0' ORDER BY publication_key ASC LIMIT ?3"
        );
        let rows = db
            .query_all_raw(raw::statement(
                db,
                sql,
                vec![
                    job_id.into(),
                    after_key.into(),
                    i64::from(limit.clamp(1, 64)).into(),
                ],
            )?)
            .await
            .context("Failed to read speech PCM replay journal")?;
        rows.iter().map(map_runtime_artifact).collect()
    }
}

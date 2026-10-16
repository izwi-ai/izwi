//! Runtime artifacts and attempt-owned stage outputs.

use super::*;

pub(crate) const MAX_STAGE_OUTPUT_ARTIFACTS: usize = 64;

pub(crate) const MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES: usize = 64;

#[derive(Debug, Clone)]
pub struct NewRuntimeArtifact {
    pub job_id: String,
    pub stage_id: Option<String>,
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
pub struct NewStageOutputArtifact {
    pub publication_key: String,
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
pub(crate) enum ReservedStageArtifactPublication {
    Published {
        asset: Box<MediaAsset>,
        artifact: RuntimeArtifact,
    },
    Existing(RuntimeArtifact),
}

pub(super) async fn get_artifact_with<C: ConnectionTrait>(
    db: &C,
    id: &str,
) -> anyhow::Result<Option<RuntimeArtifact>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            RUNTIME_ARTIFACT_BY_ID_SQL,
            vec![id.into()],
        )?)
        .await
        .context("Failed to load runtime artifact")?;
    row.as_ref().map(map_runtime_artifact).transpose()
}

pub(super) async fn stage_output_for_key_with<C: ConnectionTrait>(
    db: &C,
    lease: &StageLease,
    attempt_token: &str,
    publication_key: &str,
) -> anyhow::Result<Option<RuntimeArtifact>> {
    let sql = concat!(
        select_runtime_artifacts!(),
        " WHERE stage_id = ?1 AND producer_attempt_count = ?2",
        " AND producer_attempt_token = ?3 AND publication_key = ?4"
    );
    let row = db
        .query_one_raw(raw::statement(
            db,
            sql,
            vec![
                lease.stage_id.clone().into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.into(),
                publication_key.into(),
            ],
        )?)
        .await
        .context("Failed to load attempt-owned runtime artifact")?;
    row.as_ref().map(map_runtime_artifact).transpose()
}

pub(super) fn reserved_stage_artifact_matches(
    artifact: &RuntimeArtifact,
    media: &MediaAsset,
    expected_media: &NewMediaAsset,
    expected_artifact: &NewStageOutputArtifact,
) -> bool {
    artifact.artifact_kind == expected_artifact.artifact_kind
        && artifact.artifact_role == expected_artifact.artifact_role
        && artifact.media_asset_id.as_deref() == Some(media.id.as_str())
        && artifact.text_asset_id.is_none()
        && artifact.storage_key.is_none()
        && artifact.content_type.as_deref() == Some(expected_media.content_type.as_str())
        && artifact.filename == expected_media.filename
        && artifact.size_bytes == Some(expected_media.size_bytes)
        && artifact.sha256 == expected_media.sha256
        && artifact.metadata_json == expected_artifact.metadata_json
        && artifact.retention_policy == expected_artifact.retention_policy
        && media.asset_kind == expected_media.asset_kind
        && media.storage_namespace == expected_media.storage_namespace
        && media.content_type == expected_media.content_type
        && media.filename == expected_media.filename
        && media.size_bytes == expected_media.size_bytes
        && media.sha256 == expected_media.sha256
        && media.duration_secs == expected_media.duration_secs
        && media.sample_rate_hz == expected_media.sample_rate_hz
        && media.channel_count == expected_media.channel_count
        && media.peak_amplitude == expected_media.peak_amplitude
        && media.rms_amplitude == expected_media.rms_amplitude
        && media.source_asset_id == expected_media.source_asset_id
        && media.canonical_profile_version == expected_media.canonical_profile_version
        && media.scan_status == expected_media.scan_status
        && media.retention_policy == expected_media.retention_policy
        && media.deleted_at.is_none()
        && media.metadata_json == expected_media.metadata_json
}

pub(super) async fn active_stage_output_for_key_with<C: ConnectionTrait>(
    db: &C,
    lease: &StageLease,
    attempt_token: &str,
    publication_key: &str,
    now: i64,
) -> anyhow::Result<Option<RuntimeArtifact>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            r#"
            SELECT
                a.id,
                a.job_id,
                a.stage_id,
                a.producer_attempt_count,
                a.producer_attempt_token,
                a.publication_key,
                a.created_at,
                a.artifact_kind,
                a.artifact_role,
                a.media_asset_id,
                a.text_asset_id,
                a.storage_key,
                a.content_type,
                a.filename,
                a.size_bytes,
                a.sha256,
                a.metadata_json,
                a.retention_policy
            FROM runtime_artifacts a
            JOIN job_stages s ON s.id = a.stage_id
            JOIN runtime_jobs j ON j.id = s.job_id
            WHERE a.stage_id = ?1
              AND a.producer_attempt_count = ?2
              AND a.producer_attempt_token = ?3
              AND a.publication_key = ?4
              AND s.status = 'running'
              AND s.cancellation_state IS NULL
              AND s.worker_id = ?5
              AND s.attempt_count = ?2
              AND s.attempt_token = ?3
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?6
              AND j.status IN ('queued', 'running', 'retrying')
              AND j.cancellation_state IS NULL
            LIMIT 1
            "#,
            vec![
                lease.stage_id.clone().into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.into(),
                publication_key.into(),
                lease.worker_id.clone().into(),
                now.into(),
            ],
        )?)
        .await
        .context("Failed to load attempt-owned runtime artifact")?;
    row.as_ref().map(map_runtime_artifact).transpose()
}

pub(crate) fn validate_stage_output_artifact_retention_bounds(
    output_artifact_ids: &[String],
) -> anyhow::Result<()> {
    if output_artifact_ids.len() > MAX_STAGE_OUTPUT_ARTIFACTS {
        bail!("Stage output artifact count exceeds the limit of {MAX_STAGE_OUTPUT_ARTIFACTS}");
    }
    if output_artifact_ids
        .iter()
        .any(|artifact_id| artifact_id.len() > MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES)
    {
        bail!(
            "Stage output artifact identifier exceeds the {MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES}-byte limit"
        );
    }
    Ok(())
}

pub(crate) fn validate_stage_output_artifact_ids(
    output_artifact_ids: &[String],
) -> anyhow::Result<()> {
    validate_stage_output_artifact_retention_bounds(output_artifact_ids)?;
    let mut unique = std::collections::HashSet::with_capacity(output_artifact_ids.len());
    for artifact_id in output_artifact_ids {
        let parsed = uuid::Uuid::parse_str(artifact_id)
            .map_err(|_| anyhow!("Stage output artifact identifiers must be canonical UUIDs"))?;
        if parsed.hyphenated().to_string() != *artifact_id {
            bail!("Stage output artifact identifiers must be canonical UUIDs");
        }
        if !unique.insert(parsed) {
            bail!("Stage output artifact identifiers must be unique");
        }
    }
    Ok(())
}

impl BatchRuntimeStore {
    pub async fn create_artifact(
        &self,
        input: NewRuntimeArtifact,
    ) -> anyhow::Result<RuntimeArtifact> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO runtime_artifacts (
                id,
                job_id,
                stage_id,
                created_at,
                artifact_kind,
                artifact_role,
                media_asset_id,
                text_asset_id,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                metadata_json,
                retention_policy
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
            "#,
            vec![
                id.clone().into(),
                input.job_id.into(),
                opt_string(input.stage_id),
                now.into(),
                input.artifact_kind.as_db_value().into(),
                input.artifact_role.as_db_value().into(),
                opt_string(input.media_asset_id),
                opt_string(input.text_asset_id),
                opt_string(input.storage_key),
                opt_string(input.content_type),
                opt_string(input.filename),
                opt_u64(input.size_bytes),
                opt_string(input.sha256),
                metadata_json.into(),
                input.retention_policy.into(),
            ],
        )?)
        .await
        .context("Failed to create runtime artifact")?;

        self.get_artifact(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime artifact was not found"))
    }

    pub async fn publish_stage_output_artifact(
        &self,
        lease: &StageLease,
        input: NewStageOutputArtifact,
    ) -> anyhow::Result<Option<RuntimeArtifact>> {
        if !matches!(
            input.artifact_role,
            RuntimeArtifactRole::OutputPrimary
                | RuntimeArtifactRole::OutputIntermediate
                | RuntimeArtifactRole::Debug
        ) {
            bail!("Attempt-owned artifact publication requires an output or debug role");
        }
        let publication_key = input.publication_key.trim().to_string();
        if publication_key.is_empty() {
            bail!("Attempt-owned artifact publication requires a publication key");
        }
        let Some(attempt_token) = lease.attempt_token.as_ref() else {
            return Ok(None);
        };

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime artifact publication transaction")?;
        let now = self.now_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;
        let conflict_clause = match tx.get_database_backend() {
            DbBackend::Sqlite | DbBackend::Postgres => {
                "ON CONFLICT(stage_id, producer_attempt_token, publication_key) DO NOTHING"
            }
            backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
        };
        let insert_sql = format!(
            r#"
            INSERT INTO runtime_artifacts (
                id,
                job_id,
                stage_id,
                producer_attempt_count,
                producer_attempt_token,
                publication_key,
                created_at,
                artifact_kind,
                artifact_role,
                media_asset_id,
                text_asset_id,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                metadata_json,
                retention_policy
            )
            SELECT
                ?1,
                s.job_id,
                s.id,
                ?2,
                ?3,
                ?4,
                ?5,
                ?6,
                ?7,
                ?8,
                ?9,
                ?10,
                ?11,
                ?12,
                ?13,
                ?14,
                ?15,
                ?16
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
            {conflict_clause}
            "#
        );
        tx.execute_raw(raw::statement(
            &tx,
            insert_sql,
            vec![
                id.into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.clone().into(),
                publication_key.clone().into(),
                now.into(),
                input.artifact_kind.as_db_value().into(),
                input.artifact_role.as_db_value().into(),
                opt_string(input.media_asset_id),
                opt_string(input.text_asset_id),
                opt_string(input.storage_key),
                opt_string(input.content_type),
                opt_string(input.filename),
                opt_u64(input.size_bytes),
                opt_string(input.sha256),
                metadata_json.into(),
                input.retention_policy.into(),
                lease.stage_id.clone().into(),
                lease.worker_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to publish attempt-owned runtime artifact")?;

        let row = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT
                    a.id,
                    a.job_id,
                    a.stage_id,
                    a.producer_attempt_count,
                    a.producer_attempt_token,
                    a.publication_key,
                    a.created_at,
                    a.artifact_kind,
                    a.artifact_role,
                    a.media_asset_id,
                    a.text_asset_id,
                    a.storage_key,
                    a.content_type,
                    a.filename,
                    a.size_bytes,
                    a.sha256,
                    a.metadata_json,
                    a.retention_policy
                FROM runtime_artifacts a
                JOIN job_stages s ON s.id = a.stage_id
                JOIN runtime_jobs j ON j.id = s.job_id
                WHERE a.stage_id = ?1
                  AND a.producer_attempt_count = ?2
                  AND a.producer_attempt_token = ?3
                  AND a.publication_key = ?4
                  AND s.status = 'running'
                  AND s.cancellation_state IS NULL
                  AND s.worker_id = ?5
                  AND s.attempt_count = ?2
                  AND s.attempt_token = ?3
                  AND s.lease_expires_at IS NOT NULL
                  AND s.lease_expires_at > ?6
                  AND j.status IN ('queued', 'running', 'retrying')
                  AND j.cancellation_state IS NULL
                LIMIT 1
                "#,
                vec![
                    lease.stage_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.clone().into(),
                    publication_key.into(),
                    lease.worker_id.clone().into(),
                    now.into(),
                ],
            )?)
            .await
            .context("Failed to load attempt-owned runtime artifact")?;
        let artifact = row.as_ref().map(map_runtime_artifact).transpose()?;
        tx.commit()
            .await
            .context("Failed to commit runtime artifact publication transaction")?;
        Ok(artifact)
    }

    pub async fn get_artifact(&self, id: &str) -> anyhow::Result<Option<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                RUNTIME_ARTIFACT_BY_ID_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load runtime artifact")?;

        row.as_ref().map(map_runtime_artifact).transpose()
    }

    pub async fn list_artifacts_for_job(
        &self,
        job_id: &str,
    ) -> anyhow::Result<Vec<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to list runtime job artifacts")?;

        rows.iter().map(map_runtime_artifact).collect()
    }

    pub async fn stage_output_for_key(
        &self,
        lease: &StageLease,
        key: &str,
    ) -> anyhow::Result<Option<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let sql = concat!(
            select_runtime_artifacts!(),
            " WHERE stage_id = ?1 AND producer_attempt_token = ?2 AND publication_key = ?3"
        );
        let row = db
            .query_one_raw(raw::statement(
                db,
                sql,
                vec![
                    lease.stage_id.clone().into(),
                    opt_string(lease.attempt_token.clone()),
                    key.into(),
                ],
            )?)
            .await?;
        row.as_ref().map(map_runtime_artifact).transpose()
    }
}

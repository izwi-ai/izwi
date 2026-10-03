use crate::{db::raw, voice_defaults::DEFAULT_VOICE_PROFILE_ID};
use anyhow::{bail, Context};
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
struct RequiredSchemaTable {
    name: &'static str,
    columns: &'static [&'static str],
}

#[derive(Debug, Clone, Copy)]
struct RequiredUniqueIndex {
    table: &'static str,
    name: &'static str,
    columns: &'static [&'static str],
}

const REQUIRED_UNIQUE_INDEXES: &[RequiredUniqueIndex] = &[
    RequiredUniqueIndex {
        table: "runtime_artifacts",
        name: "idx_runtime_artifacts_attempt_publication",
        columns: &["stage_id", "producer_attempt_token", "publication_key"],
    },
    RequiredUniqueIndex {
        table: "speech_history_records",
        name: "idx_speech_history_audio_media_asset",
        columns: &["audio_media_asset_id"],
    },
];

const REQUIRED_SCHEMA_TABLES: &[RequiredSchemaTable] = &[
    RequiredSchemaTable {
        name: "runtime_admission_locks",
        columns: &["id", "lock_value"],
    },
    RequiredSchemaTable {
        name: "chat_threads",
        columns: &[
            "id",
            "title",
            "model_id",
            "system_prompt",
            "created_at",
            "updated_at",
        ],
    },
    RequiredSchemaTable {
        name: "chat_messages",
        columns: &[
            "id",
            "thread_id",
            "role",
            "content",
            "content_parts",
            "created_at",
            "tokens_generated",
            "generation_time_ms",
        ],
    },
    RequiredSchemaTable {
        name: "voice_profiles",
        columns: &[
            "id",
            "name",
            "system_prompt",
            "observational_memory_enabled",
            "created_at",
            "updated_at",
        ],
    },
    RequiredSchemaTable {
        name: "voice_sessions",
        columns: &[
            "id",
            "profile_id",
            "mode",
            "system_prompt",
            "created_at",
            "updated_at",
            "ended_at",
        ],
    },
    RequiredSchemaTable {
        name: "voice_turns",
        columns: &[
            "id",
            "session_id",
            "utterance_id",
            "utterance_seq",
            "mode",
            "status",
            "status_reason",
            "vad_end_reason",
            "user_text",
            "assistant_text",
            "assistant_raw_text",
            "language",
            "audio_duration_secs",
            "asr_model_id",
            "text_model_id",
            "tts_model_id",
            "s2s_model_id",
            "speaker",
            "created_at",
            "updated_at",
        ],
    },
    RequiredSchemaTable {
        name: "voice_observations",
        columns: &[
            "id",
            "profile_id",
            "category",
            "summary",
            "canonical_summary",
            "confidence",
            "source_turn_id",
            "source_user_text",
            "source_assistant_text",
            "times_seen",
            "created_at",
            "updated_at",
            "forgotten_at",
        ],
    },
    RequiredSchemaTable {
        name: "onboarding_state",
        columns: &["id", "completed_at", "analytics_opt_in"],
    },
    RequiredSchemaTable {
        name: "transcription_records",
        columns: &[
            "id",
            "created_at",
            "transcription_mode",
            "model_id",
            "aligner_model_id",
            "language",
            "processing_status",
            "processing_error",
            "processing_progress_json",
            "runtime_stage_id",
            "runtime_attempt_token",
            "duration_secs",
            "processing_time_ms",
            "rtf",
            "audio_mime_type",
            "audio_filename",
            "audio_storage_path",
            "transcription",
            "segments_json",
            "words_json",
            "speaker_attributed_text",
            "speaker_turns_json",
            "saa_status",
            "saa_warnings_json",
            "summary_status",
            "summary_model_id",
            "summary_text",
            "summary_error",
            "summary_updated_at",
        ],
    },
    RequiredSchemaTable {
        name: "diarization_records",
        columns: &[
            "id",
            "created_at",
            "model_id",
            "asr_model_id",
            "aligner_model_id",
            "llm_model_id",
            "processing_status",
            "processing_error",
            "min_speakers",
            "max_speakers",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "enable_llm_refinement",
            "processing_time_ms",
            "duration_secs",
            "rtf",
            "speaker_count",
            "alignment_coverage",
            "unattributed_words",
            "llm_refined",
            "asr_text",
            "raw_transcript",
            "transcript",
            "summary_status",
            "summary_model_id",
            "summary_text",
            "summary_error",
            "summary_updated_at",
            "segments_json",
            "words_json",
            "utterances_json",
            "speaker_name_overrides_json",
            "audio_mime_type",
            "audio_filename",
            "audio_storage_path",
        ],
    },
    RequiredSchemaTable {
        name: "speech_history_records",
        columns: &[
            "id",
            "created_at",
            "route_kind",
            "processing_status",
            "processing_error",
            "runtime_stage_id",
            "runtime_attempt_token",
            "model_id",
            "speaker",
            "language",
            "saved_voice_id",
            "speed",
            "input_text",
            "voice_description",
            "reference_text",
            "generation_time_ms",
            "audio_duration_secs",
            "rtf",
            "tokens_generated",
            "audio_mime_type",
            "audio_filename",
            "audio_storage_path",
            "audio_media_asset_id",
            "audio_artifact_tenant",
        ],
    },
    RequiredSchemaTable {
        name: "saved_voices",
        columns: &[
            "id",
            "created_at",
            "updated_at",
            "name",
            "reference_text",
            "audio_mime_type",
            "audio_filename",
            "audio_storage_path",
            "source_route_kind",
            "source_record_id",
            "permission_scope",
            "consent_status",
            "allowed_uses_json",
            "permission_provenance",
            "permission_revoked_at",
        ],
    },
    RequiredSchemaTable {
        name: "studio_projects",
        columns: &[
            "id",
            "created_at",
            "updated_at",
            "name",
            "source_filename",
            "source_text",
            "model_id",
            "voice_mode",
            "speaker",
            "saved_voice_id",
            "speed",
        ],
    },
    RequiredSchemaTable {
        name: "studio_project_segments",
        columns: &[
            "id",
            "project_id",
            "position",
            "text",
            "model_id",
            "voice_mode",
            "speaker",
            "saved_voice_id",
            "speech_record_id",
            "updated_at",
        ],
    },
    RequiredSchemaTable {
        name: "studio_project_folders",
        columns: &[
            "id",
            "created_at",
            "updated_at",
            "name",
            "parent_id",
            "sort_order",
        ],
    },
    RequiredSchemaTable {
        name: "studio_project_meta",
        columns: &[
            "project_id",
            "folder_id",
            "tags_json",
            "default_export_format",
            "last_render_job_id",
            "last_rendered_at",
        ],
    },
    RequiredSchemaTable {
        name: "studio_project_pronunciations",
        columns: &[
            "id",
            "project_id",
            "source_text",
            "replacement_text",
            "locale",
            "created_at",
            "updated_at",
        ],
    },
    RequiredSchemaTable {
        name: "studio_project_snapshots",
        columns: &["id", "project_id", "created_at", "label", "project_json"],
    },
    RequiredSchemaTable {
        name: "studio_project_render_jobs",
        columns: &[
            "id",
            "project_id",
            "created_at",
            "updated_at",
            "status",
            "error_message",
            "queued_segment_ids_json",
        ],
    },
    RequiredSchemaTable {
        name: "media_assets",
        columns: &[
            "id",
            "created_at",
            "updated_at",
            "asset_kind",
            "storage_namespace",
            "storage_key",
            "content_type",
            "filename",
            "size_bytes",
            "sha256",
            "duration_secs",
            "sample_rate_hz",
            "channel_count",
            "peak_amplitude",
            "rms_amplitude",
            "source_asset_id",
            "canonical_profile_version",
            "scan_status",
            "retention_policy",
            "deleted_at",
            "metadata_json",
        ],
    },
    RequiredSchemaTable {
        name: "artifact_cleanup_intents",
        columns: &[
            "id",
            "created_at",
            "updated_at",
            "available_at",
            "storage_key",
            "tenant_scope",
            "reason",
            "attempt_count",
            "last_error",
        ],
    },
    RequiredSchemaTable {
        name: "provider_write_reservations",
        columns: &[
            "write_id",
            "reservation_token",
            "created_at",
            "updated_at",
            "expires_at",
            "available_at",
            "state",
            "tenant_scope",
            "storage_namespace",
            "content_type",
            "filename",
            "expected_size_bytes",
            "expected_sha256",
            "provider_request_json",
            "storage_key",
            "cleanup_claim_token",
            "cleanup_claim_expires_at",
            "cleanup_attempt_count",
            "last_error",
        ],
    },
    RequiredSchemaTable {
        name: "text_assets",
        columns: &[
            "id",
            "created_at",
            "updated_at",
            "raw_text",
            "normalized_text",
            "language_hint",
            "character_count",
            "sha256",
            "safety_status",
            "retention_policy",
            "structure_json",
        ],
    },
    RequiredSchemaTable {
        name: "runtime_jobs",
        columns: &[
            "admission_tenant",
            "id",
            "created_at",
            "updated_at",
            "queued_at",
            "started_at",
            "finished_at",
            "job_kind",
            "status",
            "priority",
            "model_id",
            "capability",
            "route_record_kind",
            "route_record_id",
            "input_media_asset_id",
            "input_text_asset_id",
            "request_json",
            "model_snapshot_json",
            "progress_json",
            "error_code",
            "error_message",
            "attempt_count",
            "max_attempts",
            "retry_policy_json",
            "idempotency_key",
            "correlation_id",
            "cancellation_reason",
            "cancellation_state",
        ],
    },
    RequiredSchemaTable {
        name: "job_stages",
        columns: &[
            "id",
            "job_id",
            "created_at",
            "updated_at",
            "sequence",
            "stage_kind",
            "queue_class",
            "resource_hints_json",
            "resource_target",
            "required_backend",
            "required_device_class",
            "min_resource_memory_bytes",
            "resource_concurrency_weight",
            "status",
            "capability",
            "model_id",
            "worker_id",
            "lease_expires_at",
            "available_at",
            "attempt_token",
            "attempt_count",
            "max_attempts",
            "input_artifact_ids_json",
            "output_artifact_ids_json",
            "progress_json",
            "started_at",
            "finished_at",
            "error_code",
            "error_message",
            "cancellation_state",
        ],
    },
    RequiredSchemaTable {
        name: "runtime_artifacts",
        columns: &[
            "id",
            "job_id",
            "stage_id",
            "producer_attempt_count",
            "producer_attempt_token",
            "publication_key",
            "created_at",
            "artifact_kind",
            "artifact_role",
            "media_asset_id",
            "text_asset_id",
            "storage_key",
            "content_type",
            "filename",
            "size_bytes",
            "sha256",
            "metadata_json",
            "retention_policy",
        ],
    },
    RequiredSchemaTable {
        name: "idempotency_keys",
        columns: &[
            "operation",
            "idempotency_key",
            "created_at",
            "expires_at",
            "request_hash",
            "response_json",
            "runtime_job_id",
            "conflict_message",
            "metadata_json",
        ],
    },
    RequiredSchemaTable {
        name: "durable_idempotency_keys_v2",
        columns: &[
            "tenant_scope",
            "operation",
            "idempotency_key",
            "created_at",
            "updated_at",
            "expires_at",
            "digest_version",
            "request_digest",
            "state",
            "reservation_token",
            "runtime_job_id",
            "response_json",
        ],
    },
    RequiredSchemaTable {
        name: "runtime_worker_heartbeats",
        columns: &[
            "worker_id",
            "started_at",
            "last_heartbeat_at",
            "status",
            "queue_names_json",
            "instance_id",
            "registration_version",
            "registration_json",
            "heartbeat_version",
            "available_slots",
            "heartbeat_details_json",
            "current_job_id",
            "current_stage_id",
            "diagnostic_json",
        ],
    },
];

pub async fn validate_provider_managed_schema(db: &DatabaseConnection) -> anyhow::Result<()> {
    let mut missing_tables = Vec::new();
    let mut missing_columns = Vec::new();
    let mut missing_indexes = Vec::new();

    for table in REQUIRED_SCHEMA_TABLES {
        let columns = load_table_columns(db, table.name)
            .await
            .with_context(|| format!("Failed to inspect enterprise table {}", table.name))?;

        if columns.is_empty() {
            missing_tables.push(table.name);
            continue;
        }

        for required_column in table.columns {
            if !columns.contains(*required_column) {
                missing_columns.push(format!("{}.{}", table.name, required_column));
            }
        }
    }

    for index in REQUIRED_UNIQUE_INDEXES {
        let columns = load_unique_index_columns(db, index.table, index.name)
            .await
            .with_context(|| format!("Failed to inspect enterprise index {}", index.name))?;
        if columns.as_ref().is_none_or(|columns| {
            !columns
                .iter()
                .map(String::as_str)
                .eq(index.columns.iter().copied())
        }) {
            missing_indexes.push(index.name);
        }
    }

    if !missing_tables.is_empty() || !missing_columns.is_empty() || !missing_indexes.is_empty() {
        bail!(
            "Enterprise database schema is incomplete. Missing tables: [{}]. Missing columns: [{}]. Missing or invalid unique indexes: [{}]. Run the provider-managed schema setup before starting Izwi.",
            missing_tables.join(", "),
            missing_columns.join(", "),
            missing_indexes.join(", ")
        );
    }

    validate_required_seed_data(db).await?;
    Ok(())
}

async fn load_unique_index_columns(
    db: &DatabaseConnection,
    table: &'static str,
    index: &'static str,
) -> anyhow::Result<Option<Vec<String>>> {
    let backend = db.get_database_backend();
    let rows = match backend {
        DbBackend::Sqlite => {
            let indexes = db
                .query_all_raw(raw::statement_without_values(
                    db,
                    format!("PRAGMA index_list({table})"),
                ))
                .await?;
            let unique = indexes.iter().any(|row| {
                row.try_get_by_index::<String>(1).ok().as_deref() == Some(index)
                    && row.try_get_by_index::<i64>(2).ok() == Some(1)
                    && row.try_get_by_index::<i64>(4).ok() == Some(0)
            });
            if !unique {
                return Ok(None);
            }
            db.query_all_raw(raw::statement_without_values(
                db,
                format!("PRAGMA index_info({index})"),
            ))
            .await?
        }
        DbBackend::Postgres => {
            db.query_all_raw(raw::statement(
                db,
                r#"
                SELECT attribute.attname
                FROM pg_class table_class
                JOIN pg_index index_meta ON index_meta.indrelid = table_class.oid
                JOIN pg_class index_class ON index_class.oid = index_meta.indexrelid
                JOIN LATERAL unnest(index_meta.indkey) WITH ORDINALITY
                    AS index_key(attnum, ordinal) ON TRUE
                JOIN pg_attribute attribute
                    ON attribute.attrelid = table_class.oid
                   AND attribute.attnum = index_key.attnum
                WHERE table_class.oid = to_regclass(?1)
                  AND index_class.relname = ?2
                  AND index_meta.indisunique
                  AND index_meta.indpred IS NULL
                  AND index_meta.indexprs IS NULL
                  AND index_meta.indisvalid
                  AND index_meta.indisready
                ORDER BY index_key.ordinal
                "#,
                vec![table.into(), index.into()],
            )?)
            .await?
        }
        DbBackend::MySql => {
            db.query_all_raw(raw::statement(
                db,
                r#"
                SELECT COLUMN_NAME
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name = ?1
                  AND index_name = ?2
                  AND NON_UNIQUE = 0
                  AND SUB_PART IS NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM information_schema.statistics prefixed_part
                      WHERE prefixed_part.table_schema = DATABASE()
                        AND prefixed_part.table_name = ?1
                        AND prefixed_part.index_name = ?2
                        AND prefixed_part.SUB_PART IS NOT NULL
                  )
                ORDER BY SEQ_IN_INDEX
                "#,
                vec![table.into(), index.into()],
            )?)
            .await?
        }
        _ => bail!("Unsupported SeaORM database backend: {backend:?}"),
    };
    if rows.is_empty() {
        return Ok(None);
    }
    let column_index = if matches!(backend, DbBackend::Sqlite) {
        2
    } else {
        0
    };
    rows.into_iter()
        .map(|row| {
            row.try_get_by_index::<String>(column_index)
                .map(|name| name.to_ascii_lowercase())
                .map_err(Into::into)
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .map(Some)
}

async fn load_table_columns(
    db: &DatabaseConnection,
    table: &'static str,
) -> anyhow::Result<HashSet<String>> {
    let backend = db.get_database_backend();
    let rows = match backend {
        DbBackend::Sqlite => {
            db.query_all_raw(raw::statement_without_values(
                db,
                format!("PRAGMA table_info({table})"),
            ))
            .await?
        }
        DbBackend::Postgres => {
            db.query_all_raw(raw::statement(
                db,
                r#"
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = ANY (current_schemas(false))
                  AND table_name = ?1
                "#,
                vec![table.into()],
            )?)
            .await?
        }
        DbBackend::MySql => {
            db.query_all_raw(raw::statement(
                db,
                r#"
                SELECT COLUMN_NAME
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = ?1
                "#,
                vec![table.into()],
            )?)
            .await?
        }
        _ => bail!("Unsupported SeaORM database backend: {backend:?}"),
    };

    let column_index = if matches!(backend, DbBackend::Sqlite) {
        1
    } else {
        0
    };

    rows.into_iter()
        .map(|row| {
            row.try_get_by_index::<String>(column_index)
                .map(|name| name.to_ascii_lowercase())
                .map_err(Into::into)
        })
        .collect()
}

async fn validate_required_seed_data(db: &DatabaseConnection) -> anyhow::Result<()> {
    let exists = db
        .query_one_raw(raw::statement(
            db,
            "SELECT 1 FROM voice_profiles WHERE id = ?1 LIMIT 1",
            vec![DEFAULT_VOICE_PROFILE_ID.into()],
        )?)
        .await?
        .is_some();

    if !exists {
        bail!(
            "Enterprise database schema is missing required seed data: voice_profiles.{}",
            DEFAULT_VOICE_PROFILE_ID
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_unique_index_columns, validate_provider_managed_schema};
    use sea_orm::{ConnectionTrait, Database, DbBackend, Statement};

    #[tokio::test]
    async fn provider_contract_rejects_chat_threads_without_system_prompt() {
        let db = Database::connect("sqlite::memory:")
            .await
            .expect("sqlite connection");
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "CREATE TABLE chat_threads (id TEXT PRIMARY KEY, title TEXT NOT NULL, model_id TEXT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
        ))
        .await
        .expect("legacy chat_threads table");

        let error = validate_provider_managed_schema(&db)
            .await
            .expect_err("legacy provider schema should fail");
        assert!(
            error.to_string().contains("chat_threads.system_prompt"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn provider_contract_requires_exact_provider_write_request_envelopes() {
        let db = Database::connect("sqlite::memory:")
            .await
            .expect("sqlite connection");
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "CREATE TABLE provider_write_reservations (write_id TEXT PRIMARY KEY)",
        ))
        .await
        .expect("partial provider-write table");

        let error = validate_provider_managed_schema(&db)
            .await
            .expect_err("provider schema without request envelope should fail");
        assert!(
            error
                .to_string()
                .contains("provider_write_reservations.provider_request_json"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn provider_contract_requires_exact_attempt_publication_unique_index() {
        let db = Database::connect("sqlite::memory:")
            .await
            .expect("sqlite connection");
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "CREATE TABLE runtime_artifacts (stage_id TEXT, producer_attempt_token TEXT, publication_key TEXT)",
        ))
        .await
        .unwrap();
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "CREATE INDEX idx_runtime_artifacts_attempt_publication ON runtime_artifacts(stage_id, producer_attempt_token, publication_key)",
        ))
        .await
        .unwrap();
        assert!(load_unique_index_columns(
            &db,
            "runtime_artifacts",
            "idx_runtime_artifacts_attempt_publication"
        )
        .await
        .unwrap()
        .is_none());

        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "DROP INDEX idx_runtime_artifacts_attempt_publication",
        ))
        .await
        .unwrap();
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "CREATE UNIQUE INDEX idx_runtime_artifacts_attempt_publication ON runtime_artifacts(stage_id, producer_attempt_token, publication_key) WHERE publication_key IS NOT NULL",
        ))
        .await
        .unwrap();
        assert!(load_unique_index_columns(
            &db,
            "runtime_artifacts",
            "idx_runtime_artifacts_attempt_publication"
        )
        .await
        .unwrap()
        .is_none());
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "DROP INDEX idx_runtime_artifacts_attempt_publication",
        ))
        .await
        .unwrap();
        db.execute_raw(Statement::from_string(
            DbBackend::Sqlite,
            "CREATE UNIQUE INDEX idx_runtime_artifacts_attempt_publication ON runtime_artifacts(stage_id, producer_attempt_token, publication_key)",
        ))
        .await
        .unwrap();
        assert_eq!(
            load_unique_index_columns(
                &db,
                "runtime_artifacts",
                "idx_runtime_artifacts_attempt_publication"
            )
            .await
            .unwrap()
            .unwrap(),
            ["stage_id", "producer_attempt_token", "publication_key"]
        );
    }
}

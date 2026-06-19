use crate::{db::raw, voice_defaults::DEFAULT_VOICE_PROFILE_ID};
use anyhow::{bail, Context};
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
struct RequiredSchemaTable {
    name: &'static str,
    columns: &'static [&'static str],
}

const REQUIRED_SCHEMA_TABLES: &[RequiredSchemaTable] = &[
    RequiredSchemaTable {
        name: "chat_threads",
        columns: &["id", "title", "model_id", "created_at", "updated_at"],
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
            "model_id",
            "aligner_model_id",
            "language",
            "processing_status",
            "processing_error",
            "duration_secs",
            "processing_time_ms",
            "rtf",
            "audio_mime_type",
            "audio_filename",
            "audio_storage_path",
            "transcription",
            "segments_json",
            "words_json",
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
            "diarization_diagnostics_json",
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
];

pub async fn validate_provider_managed_schema(db: &DatabaseConnection) -> anyhow::Result<()> {
    let mut missing_tables = Vec::new();
    let mut missing_columns = Vec::new();

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

    if !missing_tables.is_empty() || !missing_columns.is_empty() {
        bail!(
            "Enterprise database schema is incomplete. Missing tables: [{}]. Missing columns: [{}]. Run the provider-managed schema setup before starting Izwi.",
            missing_tables.join(", "),
            missing_columns.join(", ")
        );
    }

    validate_required_seed_data(db).await?;
    Ok(())
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

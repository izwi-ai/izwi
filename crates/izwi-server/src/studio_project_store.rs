//! Persistent text-to-speech project storage backed by SQLite.

use anyhow::{Context, anyhow};
use sea_orm::{
    ConnectionTrait, DatabaseConnection, DbBackend, QueryResult, Statement, TransactionTrait, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    db::StoreDatabase,
    ids::new_uuid,
    storage_layout::{self},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioProjectVoiceMode {
    BuiltIn,
    Saved,
}

impl StudioProjectVoiceMode {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::BuiltIn => "built_in",
            Self::Saved => "saved",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "built_in" => Some(Self::BuiltIn),
            "saved" => Some(Self::Saved),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioProjectExportFormat {
    Wav,
    Mp3,
    Flac,
}

impl StudioProjectExportFormat {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "wav" => Some(Self::Wav),
            "mp3" => Some(Self::Mp3),
            "flac" => Some(Self::Flac),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioProjectRenderJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl StudioProjectRenderJobStatus {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectFolderRecord {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub parent_id: Option<String>,
    pub sort_order: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectMetaRecord {
    pub project_id: String,
    pub folder_id: Option<String>,
    pub tags: Vec<String>,
    pub default_export_format: StudioProjectExportFormat,
    pub last_render_job_id: Option<String>,
    pub last_rendered_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectPronunciationRecord {
    pub id: String,
    pub project_id: String,
    pub source_text: String,
    pub replacement_text: String,
    pub locale: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectSnapshotRecord {
    pub id: String,
    pub project_id: String,
    pub created_at: u64,
    pub label: Option<String>,
    pub project_name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectRenderJobRecord {
    pub id: String,
    pub project_id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub status: StudioProjectRenderJobStatus,
    pub error_message: Option<String>,
    pub queued_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectSummary {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub source_filename: Option<String>,
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segment_count: usize,
    pub rendered_segment_count: usize,
    pub total_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StudioProjectListCursor {
    pub updated_at: u64,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioProjectSegmentRecord {
    pub id: String,
    pub project_id: String,
    pub position: usize,
    pub text: String,
    pub model_id: Option<String>,
    pub voice_mode: Option<StudioProjectVoiceMode>,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub input_chars: usize,
    pub speech_record_id: Option<String>,
    pub updated_at: u64,
    pub generation_time_ms: Option<f64>,
    pub audio_duration_secs: Option<f64>,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioProjectRecord {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segments: Vec<StudioProjectSegmentRecord>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectSegment {
    pub position: usize,
    pub text: String,
    pub model_id: Option<String>,
    pub voice_mode: Option<StudioProjectVoiceMode>,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectRecord {
    pub name: String,
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segments: Vec<NewStudioProjectSegment>,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateStudioProjectRecord {
    pub name: Option<String>,
    pub model_id: Option<String>,
    pub voice_mode: Option<StudioProjectVoiceMode>,
    pub speaker: Option<Option<String>>,
    pub saved_voice_id: Option<Option<String>>,
    pub speed: Option<Option<f64>>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectFolderRecord {
    pub name: String,
    pub parent_id: Option<String>,
    pub sort_order: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct UpsertStudioProjectMetaRecord {
    pub folder_id: Option<Option<String>>,
    pub tags: Option<Vec<String>>,
    pub default_export_format: Option<StudioProjectExportFormat>,
    pub last_render_job_id: Option<Option<String>>,
    pub last_rendered_at: Option<Option<u64>>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectPronunciationRecord {
    pub source_text: String,
    pub replacement_text: String,
    pub locale: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectSnapshotRecord {
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectRenderJobRecord {
    pub queued_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateStudioProjectRenderJobRecord {
    pub status: Option<StudioProjectRenderJobStatus>,
    pub error_message: Option<Option<String>>,
}

#[derive(Clone)]
pub struct StudioProjectStore {
    db: StoreDatabase,
}

impl StudioProjectStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare Studio project storage layout")?;

        Ok(Self {
            db: StoreDatabase::new(db_path),
        })
    }

    pub async fn list_projects_page(
        &self,
        limit: usize,
        cursor: Option<StudioProjectListCursor>,
    ) -> anyhow::Result<(Vec<StudioProjectSummary>, Option<StudioProjectListCursor>)> {
        let db = self.db.connection().await?;
        let list_limit = i64::try_from(limit.clamp(1, 200).max(1)).unwrap_or(100);
        let page_size = usize::try_from(list_limit).unwrap_or(100);
        let fetch_limit = list_limit.saturating_add(1);

        let rows = if let Some(cursor) = cursor {
            let cursor_updated_at = i64::try_from(cursor.updated_at).unwrap_or(i64::MAX);
            query_all(
                db,
                STUDIO_PROJECT_PAGE_AFTER_CURSOR_SQL,
                vec![
                    cursor_updated_at.into(),
                    cursor.id.into(),
                    fetch_limit.into(),
                ],
                "Failed to list Studio projects page after cursor",
            )
            .await?
        } else {
            query_all(
                db,
                STUDIO_PROJECT_PAGE_SQL,
                vec![fetch_limit.into()],
                "Failed to list Studio projects page",
            )
            .await?
        };

        let mut projects = rows
            .iter()
            .map(map_project_summary_row)
            .collect::<anyhow::Result<Vec<_>>>()?;

        let has_more = projects.len() > page_size;
        if has_more {
            projects.truncate(page_size);
        }

        let next_cursor = if has_more {
            projects.last().map(|project| StudioProjectListCursor {
                updated_at: project.updated_at,
                id: project.id.clone(),
            })
        } else {
            None
        };

        Ok((projects, next_cursor))
    }

    pub async fn get_project(
        &self,
        project_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        fetch_project(db, &project_id).await
    }

    pub async fn create_project(
        &self,
        record: NewStudioProjectRecord,
    ) -> anyhow::Result<StudioProjectRecord> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project create transaction")?;
        let now = now_unix_millis_i64();
        let project_id = new_uuid();

        execute(
            &tx,
            r#"
            INSERT INTO studio_projects (
                id,
                created_at,
                updated_at,
                name,
                source_filename,
                source_text,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id,
                speed
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
            vec![
                project_id.clone().into(),
                now.into(),
                now.into(),
                record.name.into(),
                record.source_filename.into(),
                record.source_text.into(),
                record.model_id.into(),
                record.voice_mode.as_db_value().into(),
                record.speaker.into(),
                record.saved_voice_id.into(),
                record.speed.into(),
            ],
            "Failed to create Studio project",
        )
        .await?;

        for segment in record.segments {
            execute(
                &tx,
                r#"
                INSERT INTO studio_project_segments (
                    id,
                    project_id,
                    position,
                    text,
                    model_id,
                    voice_mode,
                    speaker,
                    saved_voice_id,
                    speech_record_id,
                    updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)
                "#,
                vec![
                    new_uuid().into(),
                    project_id.clone().into(),
                    usize_to_i64(segment.position).into(),
                    segment.text.into(),
                    segment.model_id.into(),
                    segment.voice_mode.map(|value| value.as_db_value()).into(),
                    segment.speaker.into(),
                    segment.saved_voice_id.into(),
                    now.into(),
                ],
                "Failed to create Studio project segment",
            )
            .await?;
        }

        tx.commit()
            .await
            .context("Failed to commit Studio project create transaction")?;
        fetch_project(db, &project_id)
            .await?
            .ok_or_else(|| anyhow!("Created Studio project was not found"))
    }

    pub async fn update_project(
        &self,
        project_id: String,
        update: UpdateStudioProjectRecord,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project update transaction")?;

        let current = query_one(
            &tx,
            r#"
            SELECT
                name,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id,
                speed
            FROM studio_projects
            WHERE id = ?1
            "#,
            vec![project_id.clone().into()],
            "Failed to load Studio project for update",
        )
        .await?
        .map(|row| {
            let voice_mode_raw: String = row.try_get_by_index(2)?;
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<String>(0)?,
                row.try_get_by_index::<Option<String>>(1)?,
                StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
                    .unwrap_or(StudioProjectVoiceMode::BuiltIn),
                row.try_get_by_index::<Option<String>>(3)?,
                row.try_get_by_index::<Option<String>>(4)?,
                row.try_get_by_index::<Option<f64>>(5)?,
            ))
        })
        .transpose()?;

        let Some((name, model_id, voice_mode, speaker, saved_voice_id, speed)) = current else {
            return Ok(None);
        };

        let next_name = update.name.unwrap_or(name);
        let next_model_id = update.model_id.or(model_id);
        let next_voice_mode = update.voice_mode.unwrap_or(voice_mode);
        let next_speaker = update.speaker.unwrap_or(speaker);
        let next_saved_voice_id = update.saved_voice_id.unwrap_or(saved_voice_id);
        let next_speed = update.speed.unwrap_or(speed);
        let now = now_unix_millis_i64();

        execute(
            &tx,
            r#"
            UPDATE studio_projects
            SET
                updated_at = ?2,
                name = ?3,
                model_id = ?4,
                voice_mode = ?5,
                speaker = ?6,
                saved_voice_id = ?7,
                speed = ?8
            WHERE id = ?1
            "#,
            vec![
                project_id.clone().into(),
                now.into(),
                next_name.into(),
                next_model_id.into(),
                next_voice_mode.as_db_value().into(),
                next_speaker.into(),
                next_saved_voice_id.into(),
                next_speed.into(),
            ],
            "Failed to update Studio project",
        )
        .await?;

        tx.commit()
            .await
            .context("Failed to commit Studio project update transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn update_segment_text(
        &self,
        project_id: String,
        segment_id: String,
        text: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment text transaction")?;
        let now = now_unix_millis_i64();

        let updated = execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                text = ?3,
                speech_record_id = NULL,
                updated_at = ?4
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                segment_id.into(),
                text.into(),
                now.into(),
            ],
            "Failed to update Studio segment text",
        )
        .await?;

        if updated == 0 {
            return Ok(None);
        }

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment text transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn update_segment_settings(
        &self,
        project_id: String,
        segment_id: String,
        model_id: String,
        voice_mode: StudioProjectVoiceMode,
        speaker: Option<String>,
        saved_voice_id: Option<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment settings transaction")?;
        let now = now_unix_millis_i64();

        let updated = execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                model_id = ?3,
                voice_mode = ?4,
                speaker = ?5,
                saved_voice_id = ?6,
                speech_record_id = NULL,
                updated_at = ?7
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                segment_id.into(),
                model_id.into(),
                voice_mode.as_db_value().into(),
                speaker.into(),
                saved_voice_id.into(),
                now.into(),
            ],
            "Failed to update Studio segment settings",
        )
        .await?;

        if updated == 0 {
            return Ok(None);
        }

        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment settings transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn update_segment_text_and_settings(
        &self,
        project_id: String,
        segment_id: String,
        text: String,
        model_id: String,
        voice_mode: StudioProjectVoiceMode,
        speaker: Option<String>,
        saved_voice_id: Option<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment text/settings transaction")?;
        let now = now_unix_millis_i64();

        let updated = execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                text = ?3,
                model_id = ?4,
                voice_mode = ?5,
                speaker = ?6,
                saved_voice_id = ?7,
                speech_record_id = NULL,
                updated_at = ?8
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                segment_id.into(),
                text.into(),
                model_id.into(),
                voice_mode.as_db_value().into(),
                speaker.into(),
                saved_voice_id.into(),
                now.into(),
            ],
            "Failed to update Studio segment text/settings",
        )
        .await?;

        if updated == 0 {
            return Ok(None);
        }

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment text/settings transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn insert_segment(
        &self,
        project_id: String,
        after_segment_id: Option<String>,
        text: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment insert transaction")?;
        let now = now_unix_millis_i64();

        let project_state = query_one(
            &tx,
            r#"
            SELECT
                model_id,
                voice_mode,
                speaker,
                saved_voice_id
            FROM studio_projects
            WHERE id = ?1
            "#,
            vec![project_id.clone().into()],
            "Failed to load Studio project segment defaults",
        )
        .await?
        .map(|row| {
            let voice_mode_raw: String = row.try_get_by_index(1)?;
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<Option<String>>(0)?,
                StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
                    .unwrap_or(StudioProjectVoiceMode::BuiltIn),
                row.try_get_by_index::<Option<String>>(2)?,
                row.try_get_by_index::<Option<String>>(3)?,
            ))
        })
        .transpose()?;

        let Some((project_model_id, project_voice_mode, project_speaker, project_saved_voice_id)) =
            project_state
        else {
            return Ok(None);
        };

        let (
            insert_position,
            inherited_model_id,
            inherited_voice_mode,
            inherited_speaker,
            inherited_saved_voice_id,
        ): (
            i64,
            Option<String>,
            StudioProjectVoiceMode,
            Option<String>,
            Option<String>,
        ) = if let Some(anchor_segment_id) = after_segment_id {
            let anchor_state = query_one(
                &tx,
                r#"
                SELECT
                    position,
                    model_id,
                    voice_mode,
                    speaker,
                    saved_voice_id
                FROM studio_project_segments
                WHERE project_id = ?1 AND id = ?2
                "#,
                vec![project_id.clone().into(), anchor_segment_id.into()],
                "Failed to load Studio segment insert anchor",
            )
            .await?
            .map(|row| {
                let voice_mode_raw: Option<String> = row.try_get_by_index(2)?;
                Ok::<_, anyhow::Error>((
                    row.try_get_by_index::<i64>(0)?,
                    row.try_get_by_index::<Option<String>>(1)?,
                    voice_mode_raw
                        .as_deref()
                        .and_then(StudioProjectVoiceMode::from_db_value),
                    row.try_get_by_index::<Option<String>>(3)?,
                    row.try_get_by_index::<Option<String>>(4)?,
                ))
            })
            .transpose()?;

            let Some((
                anchor_position,
                anchor_model_id,
                anchor_voice_mode,
                anchor_speaker,
                anchor_saved_voice_id,
            )) = anchor_state
            else {
                return Ok(None);
            };

            let resolved_voice_mode = anchor_voice_mode.unwrap_or(project_voice_mode);
            let resolved_speaker = match resolved_voice_mode {
                StudioProjectVoiceMode::BuiltIn => {
                    anchor_speaker.or_else(|| project_speaker.clone())
                }
                StudioProjectVoiceMode::Saved => None,
            };
            let resolved_saved_voice_id = match resolved_voice_mode {
                StudioProjectVoiceMode::BuiltIn => None,
                StudioProjectVoiceMode::Saved => {
                    anchor_saved_voice_id.or_else(|| project_saved_voice_id.clone())
                }
            };

            (
                anchor_position + 1,
                anchor_model_id.or_else(|| project_model_id.clone()),
                resolved_voice_mode,
                resolved_speaker,
                resolved_saved_voice_id,
            )
        } else {
            let tail_position: i64 = query_one(
                &tx,
                r#"
                SELECT MAX(position)
                FROM studio_project_segments
                WHERE project_id = ?1
                "#,
                vec![project_id.clone().into()],
                "Failed to load Studio segment tail position",
            )
            .await?
            .map(|row| row.try_get_by_index::<Option<i64>>(0))
            .transpose()?
            .flatten()
            .map(|value| value + 1)
            .unwrap_or(0);
            let resolved_speaker = match project_voice_mode {
                StudioProjectVoiceMode::BuiltIn => project_speaker.clone(),
                StudioProjectVoiceMode::Saved => None,
            };
            let resolved_saved_voice_id = match project_voice_mode {
                StudioProjectVoiceMode::BuiltIn => None,
                StudioProjectVoiceMode::Saved => project_saved_voice_id.clone(),
            };

            (
                tail_position,
                project_model_id,
                project_voice_mode,
                resolved_speaker,
                resolved_saved_voice_id,
            )
        };

        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET position = position + 1
            WHERE project_id = ?1 AND position >= ?2
            "#,
            vec![project_id.clone().into(), insert_position.into()],
            "Failed to shift Studio segments for insert",
        )
        .await?;

        execute(
            &tx,
            r#"
            INSERT INTO studio_project_segments (
                id,
                project_id,
                position,
                text,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id,
                speech_record_id,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)
            "#,
            vec![
                new_uuid().into(),
                project_id.clone().into(),
                insert_position.into(),
                text.into(),
                inherited_model_id.into(),
                inherited_voice_mode.as_db_value().into(),
                inherited_speaker.into(),
                inherited_saved_voice_id.into(),
                now.into(),
            ],
            "Failed to insert Studio segment",
        )
        .await?;

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment insert transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn split_segment(
        &self,
        project_id: String,
        segment_id: String,
        before_text: String,
        after_text: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment split transaction")?;
        let now = now_unix_millis_i64();

        let segment_state = query_one(
            &tx,
            r#"
            SELECT
                position,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id
            FROM studio_project_segments
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), segment_id.clone().into()],
            "Failed to load Studio segment for split",
        )
        .await?
        .map(|row| {
            let voice_mode_raw: Option<String> = row.try_get_by_index(2)?;
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<i64>(0)?,
                row.try_get_by_index::<Option<String>>(1)?,
                voice_mode_raw
                    .as_deref()
                    .and_then(StudioProjectVoiceMode::from_db_value),
                row.try_get_by_index::<Option<String>>(3)?,
                row.try_get_by_index::<Option<String>>(4)?,
            ))
        })
        .transpose()?;

        let Some((position, model_id, voice_mode, speaker, saved_voice_id)) = segment_state else {
            return Ok(None);
        };

        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET position = position + 1
            WHERE project_id = ?1 AND position > ?2
            "#,
            vec![project_id.clone().into(), position.into()],
            "Failed to shift Studio segments for split",
        )
        .await?;

        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                text = ?3,
                speech_record_id = NULL,
                updated_at = ?4
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                segment_id.into(),
                before_text.into(),
                now.into(),
            ],
            "Failed to update Studio split segment",
        )
        .await?;

        execute(
            &tx,
            r#"
            INSERT INTO studio_project_segments (
                id,
                project_id,
                position,
                text,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id,
                speech_record_id,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)
            "#,
            vec![
                new_uuid().into(),
                project_id.clone().into(),
                (position + 1).into(),
                after_text.into(),
                model_id.into(),
                voice_mode.map(|value| value.as_db_value()).into(),
                speaker.into(),
                saved_voice_id.into(),
                now.into(),
            ],
            "Failed to insert Studio split segment",
        )
        .await?;

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment split transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn delete_segment(
        &self,
        project_id: String,
        segment_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment delete transaction")?;
        let now = now_unix_millis_i64();

        let position = query_one(
            &tx,
            r#"
            SELECT position
            FROM studio_project_segments
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), segment_id.clone().into()],
            "Failed to load Studio segment position for delete",
        )
        .await?
        .map(|row| row.try_get_by_index::<i64>(0))
        .transpose()?;

        let Some(position) = position else {
            return Ok(None);
        };

        let deleted = execute(
            &tx,
            r#"
            DELETE FROM studio_project_segments
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), segment_id.into()],
            "Failed to delete Studio segment",
        )
        .await?;

        if deleted == 0 {
            return Ok(None);
        }

        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET position = position - 1
            WHERE project_id = ?1 AND position > ?2
            "#,
            vec![project_id.clone().into(), position.into()],
            "Failed to compact Studio segments after delete",
        )
        .await?;

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment delete transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn merge_segment_with_next(
        &self,
        project_id: String,
        segment_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment merge transaction")?;
        let now = now_unix_millis_i64();

        let current = query_one(
            &tx,
            r#"
            SELECT id, position, text
            FROM studio_project_segments
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), segment_id.into()],
            "Failed to load Studio segment for merge",
        )
        .await?
        .map(|row| {
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<String>(0)?,
                row.try_get_by_index::<i64>(1)?,
                row.try_get_by_index::<String>(2)?,
            ))
        })
        .transpose()?;
        let Some((current_id, current_position, current_text)) = current else {
            return Ok(None);
        };

        let next = query_one(
            &tx,
            r#"
            SELECT id, position, text
            FROM studio_project_segments
            WHERE project_id = ?1 AND position = ?2
            "#,
            vec![project_id.clone().into(), (current_position + 1).into()],
            "Failed to load next Studio segment for merge",
        )
        .await?
        .map(|row| {
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<String>(0)?,
                row.try_get_by_index::<i64>(1)?,
                row.try_get_by_index::<String>(2)?,
            ))
        })
        .transpose()?;
        let Some((next_id, next_position, next_text)) = next else {
            return Ok(None);
        };

        let merged_text = format!("{}\n\n{}", current_text.trim(), next_text.trim())
            .trim()
            .to_string();
        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                text = ?3,
                speech_record_id = NULL,
                updated_at = ?4
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                current_id.into(),
                merged_text.into(),
                now.into(),
            ],
            "Failed to update merged Studio segment",
        )
        .await?;

        execute(
            &tx,
            r#"
            DELETE FROM studio_project_segments
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), next_id.into()],
            "Failed to delete merged Studio segment",
        )
        .await?;

        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET position = position - 1
            WHERE project_id = ?1 AND position > ?2
            "#,
            vec![project_id.clone().into(), next_position.into()],
            "Failed to compact Studio segments after merge",
        )
        .await?;

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment merge transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn reorder_segments(
        &self,
        project_id: String,
        ordered_segment_ids: Vec<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment reorder transaction")?;

        let rows = query_all(
            &tx,
            r#"
            SELECT id
            FROM studio_project_segments
            WHERE project_id = ?1
            ORDER BY position ASC, id ASC
            "#,
            vec![project_id.clone().into()],
            "Failed to list Studio segments for reorder",
        )
        .await?;
        let existing_ids = rows
            .iter()
            .map(|row| row.try_get_by_index::<String>(0).map_err(Into::into))
            .collect::<anyhow::Result<Vec<_>>>()?;

        if existing_ids.is_empty() {
            return Ok(None);
        }
        if ordered_segment_ids.len() != existing_ids.len() {
            anyhow::bail!("Reorder request must include every project segment exactly once.");
        }

        let existing_set = existing_ids.iter().collect::<HashSet<_>>();
        let requested_set = ordered_segment_ids.iter().collect::<HashSet<_>>();
        if existing_set != requested_set {
            anyhow::bail!("Reorder request contains unknown or missing segment ids.");
        }

        let now = now_unix_millis_i64();
        for (position, segment_id) in ordered_segment_ids.iter().enumerate() {
            execute(
                &tx,
                r#"
                UPDATE studio_project_segments
                SET position = ?3, updated_at = ?4, speech_record_id = NULL
                WHERE project_id = ?1 AND id = ?2
                "#,
                vec![
                    project_id.clone().into(),
                    segment_id.clone().into(),
                    usize_to_i64(position).into(),
                    now.into(),
                ],
                "Failed to update Studio segment position",
            )
            .await?;
        }

        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment reorder transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn delete_segments(
        &self,
        project_id: String,
        segment_ids: Vec<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment batch delete transaction")?;
        let rows = query_all(
            &tx,
            r#"
            SELECT id, position
            FROM studio_project_segments
            WHERE project_id = ?1
            ORDER BY position ASC, id ASC
            "#,
            vec![project_id.clone().into()],
            "Failed to list Studio segments for batch delete",
        )
        .await?;
        let existing = rows
            .iter()
            .map(|row| {
                Ok::<_, anyhow::Error>((
                    row.try_get_by_index::<String>(0)?,
                    row.try_get_by_index::<i64>(1)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if existing.is_empty() {
            return Ok(None);
        }

        let remove_set = segment_ids.into_iter().collect::<HashSet<_>>();
        if remove_set.is_empty() {
            tx.rollback()
                .await
                .context("Failed to roll back empty Studio segment delete transaction")?;
            return fetch_project(db, &project_id).await;
        }

        let remaining_count = existing
            .iter()
            .filter(|(id, _)| !remove_set.contains(id))
            .count();
        if remaining_count == 0 {
            anyhow::bail!("A project must keep at least one segment.");
        }

        for (segment_id, _) in &existing {
            if remove_set.contains(segment_id) {
                execute(
                    &tx,
                    r#"
                    DELETE FROM studio_project_segments
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    vec![project_id.clone().into(), segment_id.clone().into()],
                    "Failed to delete Studio segment in batch",
                )
                .await?;
            }
        }

        let mut new_position = 0usize;
        for (segment_id, _) in existing {
            if remove_set.contains(segment_id.as_str()) {
                continue;
            }
            execute(
                &tx,
                r#"
                UPDATE studio_project_segments
                SET position = ?3
                WHERE project_id = ?1 AND id = ?2
                "#,
                vec![
                    project_id.clone().into(),
                    segment_id.into(),
                    usize_to_i64(new_position).into(),
                ],
                "Failed to compact Studio segment positions",
            )
            .await?;
            new_position += 1;
        }

        let now = now_unix_millis_i64();
        sync_project_source_text(&tx, &project_id).await?;
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment batch delete transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn attach_segment_record(
        &self,
        project_id: String,
        segment_id: String,
        speech_record_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio segment attach transaction")?;
        let now = now_unix_millis_i64();

        let updated = execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                speech_record_id = ?3,
                updated_at = ?4
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                segment_id.into(),
                speech_record_id.into(),
                now.into(),
            ],
            "Failed to attach speech record to Studio segment",
        )
        .await?;

        if updated == 0 {
            return Ok(None);
        }

        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment attach transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn delete_project(&self, project_id: String) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project delete transaction")?;
        execute(
            &tx,
            "DELETE FROM studio_project_segments WHERE project_id = ?1",
            vec![project_id.clone().into()],
            "Failed to delete Studio project segments",
        )
        .await?;
        let deleted = execute(
            &tx,
            "DELETE FROM studio_projects WHERE id = ?1",
            vec![project_id.into()],
            "Failed to delete Studio project",
        )
        .await?;
        tx.commit()
            .await
            .context("Failed to commit Studio project delete transaction")?;
        Ok(deleted > 0)
    }

    pub async fn list_folders(&self) -> anyhow::Result<Vec<StudioProjectFolderRecord>> {
        let db = self.db.connection().await?;
        let rows = query_all(
            db,
            r#"
            SELECT id, created_at, updated_at, name, parent_id, sort_order
            FROM studio_project_folders
            ORDER BY
                CASE WHEN parent_id IS NULL THEN 0 ELSE 1 END ASC,
                parent_id ASC,
                sort_order ASC,
                updated_at DESC,
                id DESC
            "#,
            vec![],
            "Failed to list Studio project folders",
        )
        .await?;
        rows.iter().map(map_folder_row).collect()
    }

    pub async fn create_folder(
        &self,
        record: NewStudioProjectFolderRecord,
    ) -> anyhow::Result<StudioProjectFolderRecord> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let id = new_uuid();
        execute(
            db,
            r#"
            INSERT INTO studio_project_folders (id, created_at, updated_at, name, parent_id, sort_order)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                now.into(),
                record.name.into(),
                record.parent_id.into(),
                record.sort_order.unwrap_or(0).into(),
            ],
            "Failed to create Studio project folder",
        )
        .await?;
        fetch_folder(db, &id)
            .await?
            .ok_or_else(|| anyhow!("Created Studio project folder was not found"))
    }

    pub async fn get_project_meta(
        &self,
        project_id: String,
    ) -> anyhow::Result<Option<StudioProjectMetaRecord>> {
        let db = self.db.connection().await?;
        fetch_project_meta(db, &project_id).await
    }

    pub async fn upsert_project_meta(
        &self,
        project_id: String,
        update: UpsertStudioProjectMetaRecord,
    ) -> anyhow::Result<Option<StudioProjectMetaRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project metadata transaction")?;
        let exists = project_exists(&tx, &project_id).await?;
        if !exists {
            return Ok(None);
        }

        let current = query_one(
            &tx,
            r#"
            SELECT folder_id, tags_json, default_export_format, last_render_job_id, last_rendered_at
            FROM studio_project_meta
            WHERE project_id = ?1
            "#,
            vec![project_id.clone().into()],
            "Failed to load Studio project metadata",
        )
        .await?
        .map(|row| {
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<Option<String>>(0)?,
                row.try_get_by_index::<Option<String>>(1)?,
                row.try_get_by_index::<Option<String>>(2)?,
                row.try_get_by_index::<Option<String>>(3)?,
                row.try_get_by_index::<Option<i64>>(4)?,
            ))
        })
        .transpose()?;

        let default_current_export = current
            .as_ref()
            .and_then(|(_, _, format, _, _)| format.as_deref())
            .and_then(StudioProjectExportFormat::from_db_value)
            .unwrap_or(StudioProjectExportFormat::Wav);
        let next_folder_id = update.folder_id.unwrap_or_else(|| {
            current
                .as_ref()
                .and_then(|(folder_id, _, _, _, _)| folder_id.clone())
        });
        let next_tags = update.tags.unwrap_or_else(|| {
            current
                .as_ref()
                .and_then(|(_, tags_json, _, _, _)| tags_json.clone())
                .map(|raw| parse_json_string_vec(Some(raw)))
                .unwrap_or_default()
        });
        let next_export = update
            .default_export_format
            .unwrap_or(default_current_export);
        let next_last_render_job_id = update.last_render_job_id.unwrap_or_else(|| {
            current
                .as_ref()
                .and_then(|(_, _, _, last_render_job_id, _)| last_render_job_id.clone())
        });
        let next_last_rendered_at = update.last_rendered_at.unwrap_or_else(|| {
            current
                .as_ref()
                .and_then(|(_, _, _, _, last_rendered_at)| *last_rendered_at)
                .map(i64_to_u64)
        });
        let next_last_rendered_at_i64 =
            next_last_rendered_at.and_then(|value| i64::try_from(value).ok());

        execute(
            &tx,
            r#"
            INSERT INTO studio_project_meta (
                project_id,
                folder_id,
                tags_json,
                default_export_format,
                last_render_job_id,
                last_rendered_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(project_id) DO UPDATE SET
                folder_id = excluded.folder_id,
                tags_json = excluded.tags_json,
                default_export_format = excluded.default_export_format,
                last_render_job_id = excluded.last_render_job_id,
                last_rendered_at = excluded.last_rendered_at
            "#,
            vec![
                project_id.clone().into(),
                next_folder_id.into(),
                encode_json_string_vec(next_tags.as_slice()).into(),
                next_export.as_db_value().into(),
                next_last_render_job_id.into(),
                next_last_rendered_at_i64.into(),
            ],
            "Failed to upsert Studio project metadata",
        )
        .await?;

        tx.commit()
            .await
            .context("Failed to commit Studio project metadata transaction")?;
        fetch_project_meta(db, &project_id).await
    }

    pub async fn list_project_pronunciations(
        &self,
        project_id: String,
    ) -> anyhow::Result<Vec<StudioProjectPronunciationRecord>> {
        let db = self.db.connection().await?;
        let rows = query_all(
            db,
            r#"
            SELECT id, project_id, source_text, replacement_text, locale, created_at, updated_at
            FROM studio_project_pronunciations
            WHERE project_id = ?1
            ORDER BY updated_at DESC, id DESC
            "#,
            vec![project_id.into()],
            "Failed to list Studio project pronunciations",
        )
        .await?;
        rows.iter().map(map_pronunciation_row).collect()
    }

    pub async fn create_project_pronunciation(
        &self,
        project_id: String,
        record: NewStudioProjectPronunciationRecord,
    ) -> anyhow::Result<Option<StudioProjectPronunciationRecord>> {
        let db = self.db.connection().await?;
        if !project_exists(db, &project_id).await? {
            return Ok(None);
        }

        let now = now_unix_millis_i64();
        let id = new_uuid();
        execute(
            db,
            r#"
            INSERT INTO studio_project_pronunciations (
                id, project_id, source_text, replacement_text, locale, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            vec![
                id.clone().into(),
                project_id.into(),
                record.source_text.into(),
                record.replacement_text.into(),
                record.locale.into(),
                now.into(),
                now.into(),
            ],
            "Failed to create Studio project pronunciation",
        )
        .await?;

        fetch_project_pronunciation(db, &id).await
    }

    pub async fn delete_project_pronunciation(
        &self,
        project_id: String,
        pronunciation_id: String,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let deleted = execute(
            db,
            r#"
            DELETE FROM studio_project_pronunciations
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.into(), pronunciation_id.into()],
            "Failed to delete Studio project pronunciation",
        )
        .await?;
        Ok(deleted > 0)
    }

    pub async fn list_project_snapshots(
        &self,
        project_id: String,
    ) -> anyhow::Result<Vec<StudioProjectSnapshotRecord>> {
        let db = self.db.connection().await?;
        let rows = query_all(
            db,
            STUDIO_PROJECT_SNAPSHOTS_SQL,
            vec![project_id.into()],
            "Failed to list Studio project snapshots",
        )
        .await?;
        rows.iter().map(map_snapshot_row).collect()
    }

    pub async fn create_project_snapshot(
        &self,
        project_id: String,
        record: NewStudioProjectSnapshotRecord,
    ) -> anyhow::Result<Option<StudioProjectSnapshotRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project snapshot transaction")?;
        let Some(project) = fetch_project(&tx, &project_id).await? else {
            return Ok(None);
        };
        let payload = serde_json::to_string(&project)
            .context("Failed to serialize project snapshot payload")?;
        let now = now_unix_millis_i64();
        let snapshot_id = new_uuid();
        execute(
            &tx,
            r#"
            INSERT INTO studio_project_snapshots (id, project_id, created_at, label, project_json)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            vec![
                snapshot_id.clone().into(),
                project_id.into(),
                now.into(),
                record.label.into(),
                payload.into(),
            ],
            "Failed to create Studio project snapshot",
        )
        .await?;
        tx.commit()
            .await
            .context("Failed to commit Studio project snapshot transaction")?;

        fetch_snapshot(db, &snapshot_id).await
    }

    pub async fn restore_project_snapshot(
        &self,
        project_id: String,
        snapshot_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project snapshot restore transaction")?;
        let snapshot_json = query_one(
            &tx,
            r#"
            SELECT project_json
            FROM studio_project_snapshots
            WHERE id = ?1 AND project_id = ?2
            "#,
            vec![snapshot_id.into(), project_id.clone().into()],
            "Failed to load Studio project snapshot payload",
        )
        .await?
        .map(|row| row.try_get_by_index::<String>(0))
        .transpose()?;
        let Some(snapshot_json) = snapshot_json else {
            return Ok(None);
        };

        let snapshot: StudioProjectRecord = serde_json::from_str(snapshot_json.as_str())
            .context("Failed to decode project snapshot payload")?;
        let now = now_unix_millis_i64();
        execute(
            &tx,
            r#"
            UPDATE studio_projects
            SET
                updated_at = ?2,
                name = ?3,
                source_filename = ?4,
                source_text = ?5,
                model_id = ?6,
                voice_mode = ?7,
                speaker = ?8,
                saved_voice_id = ?9,
                speed = ?10
            WHERE id = ?1
            "#,
            vec![
                project_id.clone().into(),
                now.into(),
                snapshot.name.into(),
                snapshot.source_filename.into(),
                snapshot.source_text.into(),
                snapshot.model_id.into(),
                snapshot.voice_mode.as_db_value().into(),
                snapshot.speaker.into(),
                snapshot.saved_voice_id.into(),
                snapshot.speed.into(),
            ],
            "Failed to restore Studio project snapshot project row",
        )
        .await?;

        execute(
            &tx,
            "DELETE FROM studio_project_segments WHERE project_id = ?1",
            vec![project_id.clone().into()],
            "Failed to clear Studio project segments for snapshot restore",
        )
        .await?;
        for segment in snapshot.segments {
            execute(
                &tx,
                r#"
                INSERT INTO studio_project_segments (
                    id,
                    project_id,
                    position,
                    text,
                    model_id,
                    voice_mode,
                    speaker,
                    saved_voice_id,
                    speech_record_id,
                    updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
                vec![
                    segment.id.into(),
                    project_id.clone().into(),
                    usize_to_i64(segment.position).into(),
                    segment.text.into(),
                    segment.model_id.into(),
                    segment.voice_mode.map(|value| value.as_db_value()).into(),
                    segment.speaker.into(),
                    segment.saved_voice_id.into(),
                    segment.speech_record_id.into(),
                    now.into(),
                ],
                "Failed to restore Studio project snapshot segment",
            )
            .await?;
        }
        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio project snapshot restore transaction")?;
        fetch_project(db, &project_id).await
    }

    pub async fn list_project_render_jobs(
        &self,
        project_id: String,
    ) -> anyhow::Result<Vec<StudioProjectRenderJobRecord>> {
        let db = self.db.connection().await?;
        let rows = query_all(
            db,
            STUDIO_PROJECT_RENDER_JOBS_SQL,
            vec![project_id.into()],
            "Failed to list Studio project render jobs",
        )
        .await?;
        rows.iter().map(map_render_job_row).collect()
    }

    pub async fn create_project_render_job(
        &self,
        project_id: String,
        record: NewStudioProjectRenderJobRecord,
    ) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project render job transaction")?;
        let exists = project_exists(&tx, &project_id).await?;
        if !exists {
            return Ok(None);
        }
        let now = now_unix_millis_i64();
        let job_id = new_uuid();
        execute(
            &tx,
            r#"
            INSERT INTO studio_project_render_jobs (
                id,
                project_id,
                created_at,
                updated_at,
                status,
                error_message,
                queued_segment_ids_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6)
            "#,
            vec![
                job_id.clone().into(),
                project_id.clone().into(),
                now.into(),
                now.into(),
                StudioProjectRenderJobStatus::Queued.as_db_value().into(),
                encode_json_string_vec(record.queued_segment_ids.as_slice()).into(),
            ],
            "Failed to create Studio project render job",
        )
        .await?;

        upsert_last_render_job(&tx, &project_id, &job_id, None).await?;

        tx.commit()
            .await
            .context("Failed to commit Studio project render job transaction")?;
        fetch_render_job(db, &project_id, &job_id).await
    }

    pub async fn update_project_render_job(
        &self,
        project_id: String,
        job_id: String,
        update: UpdateStudioProjectRenderJobRecord,
    ) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start Studio project render job update transaction")?;
        let current = query_one(
            &tx,
            r#"
            SELECT status, error_message
            FROM studio_project_render_jobs
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), job_id.clone().into()],
            "Failed to load Studio project render job for update",
        )
        .await?
        .map(|row| {
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<String>(0)?,
                row.try_get_by_index::<Option<String>>(1)?,
            ))
        })
        .transpose()?;
        let Some((current_status, current_error)) = current else {
            return Ok(None);
        };

        let next_status = update
            .status
            .or_else(|| StudioProjectRenderJobStatus::from_db_value(current_status.as_str()))
            .unwrap_or(StudioProjectRenderJobStatus::Queued);
        let next_error = update.error_message.unwrap_or(current_error);
        let now = now_unix_millis_i64();
        execute(
            &tx,
            r#"
            UPDATE studio_project_render_jobs
            SET updated_at = ?3, status = ?4, error_message = ?5
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                project_id.clone().into(),
                job_id.clone().into(),
                now.into(),
                next_status.as_db_value().into(),
                next_error.into(),
            ],
            "Failed to update Studio project render job",
        )
        .await?;

        if next_status == StudioProjectRenderJobStatus::Completed {
            upsert_last_render_job(&tx, &project_id, &job_id, Some(now)).await?;
        }

        tx.commit()
            .await
            .context("Failed to commit Studio project render job update transaction")?;
        fetch_render_job(db, &project_id, &job_id).await
    }
}

const STUDIO_PROJECT_PAGE_SQL: &str = r#"
    SELECT
        p.id,
        p.created_at,
        p.updated_at,
        p.name,
        p.source_filename,
        p.model_id,
        p.voice_mode,
        p.speaker,
        p.saved_voice_id,
        p.speed,
        COUNT(s.id) AS segment_count,
        COALESCE(SUM(CASE WHEN s.speech_record_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS rendered_segment_count,
        COALESCE(SUM(LENGTH(s.text)), 0) AS total_chars
    FROM studio_projects p
    LEFT JOIN studio_project_segments s ON s.project_id = p.id
    GROUP BY
        p.id,
        p.created_at,
        p.updated_at,
        p.name,
        p.source_filename,
        p.model_id,
        p.voice_mode,
        p.speaker,
        p.saved_voice_id,
        p.speed
    ORDER BY p.updated_at DESC, p.id DESC
    LIMIT ?1
"#;

const STUDIO_PROJECT_PAGE_AFTER_CURSOR_SQL: &str = r#"
    SELECT
        p.id,
        p.created_at,
        p.updated_at,
        p.name,
        p.source_filename,
        p.model_id,
        p.voice_mode,
        p.speaker,
        p.saved_voice_id,
        p.speed,
        COUNT(s.id) AS segment_count,
        COALESCE(SUM(CASE WHEN s.speech_record_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS rendered_segment_count,
        COALESCE(SUM(LENGTH(s.text)), 0) AS total_chars
    FROM studio_projects p
    LEFT JOIN studio_project_segments s ON s.project_id = p.id
    WHERE p.updated_at < ?1 OR (p.updated_at = ?1 AND p.id < ?2)
    GROUP BY
        p.id,
        p.created_at,
        p.updated_at,
        p.name,
        p.source_filename,
        p.model_id,
        p.voice_mode,
        p.speaker,
        p.saved_voice_id,
        p.speed
    ORDER BY p.updated_at DESC, p.id DESC
    LIMIT ?3
"#;

const STUDIO_PROJECT_BY_ID_SQL: &str = r#"
    SELECT
        id,
        created_at,
        updated_at,
        name,
        source_filename,
        source_text,
        model_id,
        voice_mode,
        speaker,
        saved_voice_id,
        speed
    FROM studio_projects
    WHERE id = ?1
"#;

const STUDIO_PROJECT_SEGMENTS_SQL: &str = r#"
    SELECT
        s.id,
        s.project_id,
        s.position,
        s.text,
        s.model_id,
        s.voice_mode,
        s.speaker,
        s.saved_voice_id,
        s.speech_record_id,
        s.updated_at,
        h.generation_time_ms,
        h.audio_duration_secs,
        h.audio_filename
    FROM studio_project_segments s
    LEFT JOIN speech_history_records h
        ON h.id = s.speech_record_id
       AND h.route_kind = 'text_to_speech'
    WHERE s.project_id = ?1
    ORDER BY s.position ASC, s.id ASC
"#;

const STUDIO_PROJECT_META_SQL: &str = r#"
    SELECT
        project_id,
        folder_id,
        tags_json,
        default_export_format,
        last_render_job_id,
        last_rendered_at
    FROM studio_project_meta
    WHERE project_id = ?1
"#;

const STUDIO_PROJECT_SNAPSHOTS_SQL: &str = r#"
    SELECT
        s.id,
        s.project_id,
        s.created_at,
        s.label,
        COALESCE(json_extract(s.project_json, '$.name'), p.name) AS project_name
    FROM studio_project_snapshots s
    LEFT JOIN studio_projects p ON p.id = s.project_id
    WHERE s.project_id = ?1
    ORDER BY s.created_at DESC, s.id DESC
"#;

const STUDIO_PROJECT_SNAPSHOT_BY_ID_SQL: &str = r#"
    SELECT
        s.id,
        s.project_id,
        s.created_at,
        s.label,
        COALESCE(json_extract(s.project_json, '$.name'), p.name) AS project_name
    FROM studio_project_snapshots s
    LEFT JOIN studio_projects p ON p.id = s.project_id
    WHERE s.id = ?1
"#;

const STUDIO_PROJECT_RENDER_JOBS_SQL: &str = r#"
    SELECT
        id,
        project_id,
        created_at,
        updated_at,
        status,
        error_message,
        queued_segment_ids_json
    FROM studio_project_render_jobs
    WHERE project_id = ?1
    ORDER BY created_at DESC, id DESC
"#;

const STUDIO_PROJECT_RENDER_JOB_BY_ID_SQL: &str = r#"
    SELECT
        id,
        project_id,
        created_at,
        updated_at,
        status,
        error_message,
        queued_segment_ids_json
    FROM studio_project_render_jobs
    WHERE project_id = ?1 AND id = ?2
"#;

async fn fetch_project<C>(db: &C, project_id: &str) -> anyhow::Result<Option<StudioProjectRecord>>
where
    C: ConnectionTrait,
{
    let project = query_one(
        db,
        STUDIO_PROJECT_BY_ID_SQL,
        vec![project_id.into()],
        "Failed to load Studio project",
    )
    .await?
    .as_ref()
    .map(map_project_row)
    .transpose()?;

    let Some(mut project) = project else {
        return Ok(None);
    };

    let rows = query_all(
        db,
        STUDIO_PROJECT_SEGMENTS_SQL,
        vec![project_id.into()],
        "Failed to load Studio project segments",
    )
    .await?;
    project.segments = rows
        .iter()
        .map(map_segment_row)
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(Some(project))
}

async fn fetch_folder(
    db: &DatabaseConnection,
    folder_id: &str,
) -> anyhow::Result<Option<StudioProjectFolderRecord>> {
    query_one(
        db,
        r#"
        SELECT id, created_at, updated_at, name, parent_id, sort_order
        FROM studio_project_folders
        WHERE id = ?1
        "#,
        vec![folder_id.into()],
        "Failed to load Studio project folder",
    )
    .await?
    .as_ref()
    .map(map_folder_row)
    .transpose()
}

async fn fetch_project_meta<C>(
    db: &C,
    project_id: &str,
) -> anyhow::Result<Option<StudioProjectMetaRecord>>
where
    C: ConnectionTrait,
{
    query_one(
        db,
        STUDIO_PROJECT_META_SQL,
        vec![project_id.into()],
        "Failed to load Studio project metadata",
    )
    .await?
    .as_ref()
    .map(map_meta_row)
    .transpose()
}

async fn fetch_project_pronunciation(
    db: &DatabaseConnection,
    pronunciation_id: &str,
) -> anyhow::Result<Option<StudioProjectPronunciationRecord>> {
    query_one(
        db,
        r#"
        SELECT id, project_id, source_text, replacement_text, locale, created_at, updated_at
        FROM studio_project_pronunciations
        WHERE id = ?1
        "#,
        vec![pronunciation_id.into()],
        "Failed to load Studio project pronunciation",
    )
    .await?
    .as_ref()
    .map(map_pronunciation_row)
    .transpose()
}

async fn fetch_snapshot(
    db: &DatabaseConnection,
    snapshot_id: &str,
) -> anyhow::Result<Option<StudioProjectSnapshotRecord>> {
    query_one(
        db,
        STUDIO_PROJECT_SNAPSHOT_BY_ID_SQL,
        vec![snapshot_id.into()],
        "Failed to load Studio project snapshot",
    )
    .await?
    .as_ref()
    .map(map_snapshot_row)
    .transpose()
}

async fn fetch_render_job<C>(
    db: &C,
    project_id: &str,
    job_id: &str,
) -> anyhow::Result<Option<StudioProjectRenderJobRecord>>
where
    C: ConnectionTrait,
{
    query_one(
        db,
        STUDIO_PROJECT_RENDER_JOB_BY_ID_SQL,
        vec![project_id.into(), job_id.into()],
        "Failed to load Studio project render job",
    )
    .await?
    .as_ref()
    .map(map_render_job_row)
    .transpose()
}

async fn project_exists<C>(db: &C, project_id: &str) -> anyhow::Result<bool>
where
    C: ConnectionTrait,
{
    Ok(query_one(
        db,
        "SELECT 1 FROM studio_projects WHERE id = ?1",
        vec![project_id.into()],
        "Failed to check Studio project existence",
    )
    .await?
    .is_some())
}

async fn touch_project<C>(db: &C, project_id: &str, updated_at: i64) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    execute(
        db,
        "UPDATE studio_projects SET updated_at = ?2 WHERE id = ?1",
        vec![project_id.into(), updated_at.into()],
        "Failed to touch Studio project",
    )
    .await?;
    Ok(())
}

async fn sync_project_source_text<C>(db: &C, project_id: &str) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    let rows = query_all(
        db,
        r#"
        SELECT text
        FROM studio_project_segments
        WHERE project_id = ?1
        ORDER BY position ASC, id ASC
        "#,
        vec![project_id.into()],
        "Failed to load Studio segment text for source sync",
    )
    .await?;

    let segment_texts = rows
        .iter()
        .map(|row| row.try_get_by_index::<String>(0).map_err(Into::into))
        .collect::<anyhow::Result<Vec<_>>>()?;

    execute(
        db,
        "UPDATE studio_projects SET source_text = ?2 WHERE id = ?1",
        vec![project_id.into(), segment_texts.join("\n\n").into()],
        "Failed to sync Studio project source text",
    )
    .await?;
    Ok(())
}

async fn upsert_last_render_job<C>(
    db: &C,
    project_id: &str,
    job_id: &str,
    last_rendered_at: Option<i64>,
) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    if let Some(last_rendered_at) = last_rendered_at {
        execute(
            db,
            r#"
            INSERT INTO studio_project_meta (
                project_id,
                folder_id,
                tags_json,
                default_export_format,
                last_render_job_id,
                last_rendered_at
            )
            VALUES (
                ?1,
                (SELECT folder_id FROM studio_project_meta WHERE project_id = ?1),
                COALESCE((SELECT tags_json FROM studio_project_meta WHERE project_id = ?1), '[]'),
                COALESCE((SELECT default_export_format FROM studio_project_meta WHERE project_id = ?1), 'wav'),
                ?2,
                ?3
            )
            ON CONFLICT(project_id) DO UPDATE SET
                last_render_job_id = excluded.last_render_job_id,
                last_rendered_at = excluded.last_rendered_at
            "#,
            vec![project_id.into(), job_id.into(), last_rendered_at.into()],
            "Failed to update Studio project completed render metadata",
        )
        .await?;
    } else {
        execute(
            db,
            r#"
            INSERT INTO studio_project_meta (
                project_id,
                folder_id,
                tags_json,
                default_export_format,
                last_render_job_id,
                last_rendered_at
            )
            VALUES (
                ?1,
                (SELECT folder_id FROM studio_project_meta WHERE project_id = ?1),
                COALESCE((SELECT tags_json FROM studio_project_meta WHERE project_id = ?1), '[]'),
                COALESCE((SELECT default_export_format FROM studio_project_meta WHERE project_id = ?1), 'wav'),
                ?2,
                (SELECT last_rendered_at FROM studio_project_meta WHERE project_id = ?1)
            )
            ON CONFLICT(project_id) DO UPDATE SET
                last_render_job_id = excluded.last_render_job_id
            "#,
            vec![project_id.into(), job_id.into()],
            "Failed to update Studio project render metadata",
        )
        .await?;
    }
    Ok(())
}

async fn execute<C>(
    db: &C,
    sql: impl Into<String>,
    values: Vec<Value>,
    context: &'static str,
) -> anyhow::Result<u64>
where
    C: ConnectionTrait,
{
    let result = db
        .execute_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            sql,
            values,
        ))
        .await
        .context(context)?;
    Ok(result.rows_affected())
}

async fn query_one<C>(
    db: &C,
    sql: impl Into<String>,
    values: Vec<Value>,
    context: &'static str,
) -> anyhow::Result<Option<QueryResult>>
where
    C: ConnectionTrait,
{
    db.query_one_raw(Statement::from_sql_and_values(
        DbBackend::Sqlite,
        sql,
        values,
    ))
    .await
    .context(context)
}

async fn query_all<C>(
    db: &C,
    sql: impl Into<String>,
    values: Vec<Value>,
    context: &'static str,
) -> anyhow::Result<Vec<QueryResult>>
where
    C: ConnectionTrait,
{
    db.query_all_raw(Statement::from_sql_and_values(
        DbBackend::Sqlite,
        sql,
        values,
    ))
    .await
    .context(context)
}

fn map_project_summary_row(row: &QueryResult) -> anyhow::Result<StudioProjectSummary> {
    let voice_mode_raw: String = row.try_get_by_index(6)?;
    let segment_count = row.try_get_by_index::<i64>(10)?;
    let rendered_segment_count = row.try_get_by_index::<i64>(11)?;
    let total_chars = row.try_get_by_index::<i64>(12)?;

    Ok(StudioProjectSummary {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        updated_at: i64_to_u64(row.try_get_by_index(2)?),
        name: row.try_get_by_index(3)?,
        source_filename: row.try_get_by_index(4)?,
        model_id: row.try_get_by_index(5)?,
        voice_mode: StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(StudioProjectVoiceMode::BuiltIn),
        speaker: row.try_get_by_index(7)?,
        saved_voice_id: row.try_get_by_index(8)?,
        speed: row.try_get_by_index(9)?,
        segment_count: i64_to_usize(segment_count).unwrap_or_default(),
        rendered_segment_count: i64_to_usize(rendered_segment_count).unwrap_or_default(),
        total_chars: i64_to_usize(total_chars).unwrap_or_default(),
    })
}

fn map_project_row(row: &QueryResult) -> anyhow::Result<StudioProjectRecord> {
    let voice_mode_raw: String = row.try_get_by_index(7)?;
    Ok(StudioProjectRecord {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        updated_at: i64_to_u64(row.try_get_by_index(2)?),
        name: row.try_get_by_index(3)?,
        source_filename: row.try_get_by_index(4)?,
        source_text: row.try_get_by_index(5)?,
        model_id: row.try_get_by_index(6)?,
        voice_mode: StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(StudioProjectVoiceMode::BuiltIn),
        speaker: row.try_get_by_index(8)?,
        saved_voice_id: row.try_get_by_index(9)?,
        speed: row.try_get_by_index(10)?,
        segments: Vec::new(),
    })
}

fn map_segment_row(row: &QueryResult) -> anyhow::Result<StudioProjectSegmentRecord> {
    let text: String = row.try_get_by_index(3)?;
    let voice_mode_raw: Option<String> = row.try_get_by_index(5)?;
    Ok(StudioProjectSegmentRecord {
        id: row.try_get_by_index(0)?,
        project_id: row.try_get_by_index(1)?,
        position: i64_to_usize(row.try_get_by_index(2)?).unwrap_or_default(),
        input_chars: text.chars().count(),
        text,
        model_id: row.try_get_by_index(4)?,
        voice_mode: voice_mode_raw
            .as_deref()
            .and_then(StudioProjectVoiceMode::from_db_value),
        speaker: row.try_get_by_index(6)?,
        saved_voice_id: row.try_get_by_index(7)?,
        speech_record_id: row.try_get_by_index(8)?,
        updated_at: i64_to_u64(row.try_get_by_index(9)?),
        generation_time_ms: row.try_get_by_index(10)?,
        audio_duration_secs: row.try_get_by_index(11)?,
        audio_filename: row.try_get_by_index(12)?,
    })
}

fn map_folder_row(row: &QueryResult) -> anyhow::Result<StudioProjectFolderRecord> {
    Ok(StudioProjectFolderRecord {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        updated_at: i64_to_u64(row.try_get_by_index(2)?),
        name: row.try_get_by_index(3)?,
        parent_id: row.try_get_by_index(4)?,
        sort_order: row.try_get_by_index(5)?,
    })
}

fn map_meta_row(row: &QueryResult) -> anyhow::Result<StudioProjectMetaRecord> {
    let tags_json = row.try_get_by_index::<Option<String>>(2)?;
    let export_format_raw = row.try_get_by_index::<Option<String>>(3)?;
    let last_rendered_at = row.try_get_by_index::<Option<i64>>(5)?;
    Ok(StudioProjectMetaRecord {
        project_id: row.try_get_by_index(0)?,
        folder_id: row.try_get_by_index(1)?,
        tags: parse_json_string_vec(tags_json),
        default_export_format: export_format_raw
            .as_deref()
            .and_then(StudioProjectExportFormat::from_db_value)
            .unwrap_or(StudioProjectExportFormat::Wav),
        last_render_job_id: row.try_get_by_index(4)?,
        last_rendered_at: last_rendered_at.map(i64_to_u64),
    })
}

fn map_pronunciation_row(row: &QueryResult) -> anyhow::Result<StudioProjectPronunciationRecord> {
    Ok(StudioProjectPronunciationRecord {
        id: row.try_get_by_index(0)?,
        project_id: row.try_get_by_index(1)?,
        source_text: row.try_get_by_index(2)?,
        replacement_text: row.try_get_by_index(3)?,
        locale: row.try_get_by_index(4)?,
        created_at: i64_to_u64(row.try_get_by_index(5)?),
        updated_at: i64_to_u64(row.try_get_by_index(6)?),
    })
}

fn map_snapshot_row(row: &QueryResult) -> anyhow::Result<StudioProjectSnapshotRecord> {
    Ok(StudioProjectSnapshotRecord {
        id: row.try_get_by_index(0)?,
        project_id: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?),
        label: row.try_get_by_index(3)?,
        project_name: row.try_get_by_index(4)?,
    })
}

fn map_render_job_row(row: &QueryResult) -> anyhow::Result<StudioProjectRenderJobRecord> {
    let status_raw: String = row.try_get_by_index(4)?;
    let queued_segment_ids_json = row.try_get_by_index::<Option<String>>(6)?;
    Ok(StudioProjectRenderJobRecord {
        id: row.try_get_by_index(0)?,
        project_id: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?),
        updated_at: i64_to_u64(row.try_get_by_index(3)?),
        status: StudioProjectRenderJobStatus::from_db_value(status_raw.as_str())
            .unwrap_or(StudioProjectRenderJobStatus::Queued),
        error_message: row.try_get_by_index(5)?,
        queued_segment_ids: parse_json_string_vec(queued_segment_ids_json),
    })
}

fn parse_json_string_vec(raw: Option<String>) -> Vec<String> {
    raw.and_then(|value| serde_json::from_str::<Vec<String>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn encode_json_string_vec(values: &[String]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".to_string())
}

fn now_unix_millis_i64() -> i64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn i64_to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

fn i64_to_usize(value: i64) -> Option<usize> {
    usize::try_from(value).ok()
}

fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        NewStudioProjectRecord, NewStudioProjectSegment, StudioProjectStore,
        StudioProjectVoiceMode, UpdateStudioProjectRecord,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_env_root(prefix: &str) -> PathBuf {
        let mut root = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        root.push(format!("izwi-{prefix}-{nonce}"));
        root
    }

    async fn initialize_test_store(db_path: PathBuf, media_dir: PathBuf) -> StudioProjectStore {
        let store =
            StudioProjectStore::initialize_at(db_path, media_dir).expect("store should initialize");
        store
            .db
            .connection()
            .await
            .expect("schema should initialize");
        store
    }

    #[tokio::test]
    async fn create_and_update_project_round_trips_segments() {
        let root = test_env_root("studio-project-store");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store = initialize_test_store(db_path, media_dir).await;
        let created = store
            .create_project(NewStudioProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Hello world. Another sentence.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: StudioProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.1),
                segments: vec![
                    NewStudioProjectSegment {
                        position: 0,
                        text: "Hello world.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                    NewStudioProjectSegment {
                        position: 1,
                        text: "Another sentence.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                ],
            })
            .await
            .expect("project should be created");

        assert_eq!(created.segments.len(), 2);
        assert_eq!(created.speaker.as_deref(), Some("Vivian"));

        let updated = store
            .update_project(
                created.id.clone(),
                UpdateStudioProjectRecord {
                    voice_mode: Some(StudioProjectVoiceMode::Saved),
                    speaker: Some(None),
                    saved_voice_id: Some(Some("voice-1".to_string())),
                    ..UpdateStudioProjectRecord::default()
                },
            )
            .await
            .expect("update should succeed")
            .expect("project should exist");

        assert_eq!(updated.voice_mode, StudioProjectVoiceMode::Saved);
        assert_eq!(updated.saved_voice_id.as_deref(), Some("voice-1"));
        assert_eq!(updated.speaker, None);

        let segment = updated.segments.first().expect("segment should exist");
        let refreshed = store
            .update_segment_text(
                updated.id.clone(),
                segment.id.clone(),
                "Updated line.".to_string(),
            )
            .await
            .expect("segment update should succeed")
            .expect("project should exist");

        assert_eq!(refreshed.segments[0].text, "Updated line.");
        assert_eq!(refreshed.segments[0].speech_record_id, None);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn split_and_delete_segment_reorders_project_blocks() {
        let root = test_env_root("studio-project-segment-ops");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store = initialize_test_store(db_path, media_dir).await;

        let created = store
            .create_project(NewStudioProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Hello world. Another sentence.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: StudioProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.0),
                segments: vec![
                    NewStudioProjectSegment {
                        position: 0,
                        text: "Hello world. Another sentence.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                    NewStudioProjectSegment {
                        position: 1,
                        text: "Closing line.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                ],
            })
            .await
            .expect("project should be created");

        let first_segment = created.segments.first().expect("segment should exist");
        let split = store
            .split_segment(
                created.id.clone(),
                first_segment.id.clone(),
                "Hello world.".to_string(),
                "Another sentence.".to_string(),
            )
            .await
            .expect("split should succeed")
            .expect("project should exist");

        assert_eq!(split.segments.len(), 3);
        assert_eq!(split.segments[0].text, "Hello world.");
        assert_eq!(split.segments[1].text, "Another sentence.");
        assert_eq!(split.segments[2].text, "Closing line.");
        assert_eq!(
            split.source_text,
            "Hello world.\n\nAnother sentence.\n\nClosing line."
        );

        let deleted = store
            .delete_segment(split.id.clone(), split.segments[1].id.clone())
            .await
            .expect("delete should succeed")
            .expect("project should exist");

        assert_eq!(deleted.segments.len(), 2);
        assert_eq!(deleted.segments[0].position, 0);
        assert_eq!(deleted.segments[1].position, 1);
        assert_eq!(deleted.segments[0].text, "Hello world.");
        assert_eq!(deleted.segments[1].text, "Closing line.");
        assert_eq!(deleted.source_text, "Hello world.\n\nClosing line.");

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn insert_segment_places_block_after_anchor_and_inherits_render_settings() {
        let root = test_env_root("studio-project-insert-segment");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store = initialize_test_store(db_path, media_dir).await;

        let created = store
            .create_project(NewStudioProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Intro block.\n\nClosing block.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: StudioProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.0),
                segments: vec![
                    NewStudioProjectSegment {
                        position: 0,
                        text: "Intro block.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                    NewStudioProjectSegment {
                        position: 1,
                        text: "Closing block.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                ],
            })
            .await
            .expect("project should be created");

        let anchor_segment = created.segments[0].clone();
        let inserted = store
            .insert_segment(
                created.id.clone(),
                Some(anchor_segment.id.clone()),
                "Middle block.".to_string(),
            )
            .await
            .expect("insert should succeed")
            .expect("project should exist");

        assert_eq!(inserted.segments.len(), 3);
        assert_eq!(inserted.segments[0].text, "Intro block.");
        assert_eq!(inserted.segments[1].text, "Middle block.");
        assert_eq!(inserted.segments[2].text, "Closing block.");
        assert_eq!(inserted.segments[1].position, 1);
        assert_eq!(inserted.segments[2].position, 2);
        assert_eq!(inserted.segments[1].model_id.as_deref(), Some("Qwen3-TTS"));
        assert_eq!(
            inserted.segments[1].voice_mode,
            Some(StudioProjectVoiceMode::BuiltIn)
        );
        assert_eq!(inserted.segments[1].speaker.as_deref(), Some("Vivian"));
        assert_eq!(inserted.segments[1].saved_voice_id, None);
        assert_eq!(
            inserted.source_text,
            "Intro block.\n\nMiddle block.\n\nClosing block."
        );

        let _ = std::fs::remove_dir_all(root);
    }
}

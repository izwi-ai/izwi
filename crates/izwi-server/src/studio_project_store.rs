//! Persistent text-to-speech project storage backed by SQLite.

use anyhow::{anyhow, Context};
use sea_orm::{ConnectionTrait, DatabaseConnection, QueryResult, TransactionTrait, Value};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::StoreError;
use crate::{
    db::{raw, write_transaction_options, StoreDatabase},
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

/// Where a segment's audio stands. `stale` audio was rendered from other
/// text, voice, speed or pronunciations than the segment has now.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioSegmentRenderState {
    #[default]
    NotRendered,
    Rendered,
    Stale,
    Queued,
    Rendering,
    Failed,
}

impl StudioSegmentRenderState {
    /// Whether "render what's left" includes this segment.
    pub const fn needs_render(self) -> bool {
        matches!(self, Self::NotRendered | Self::Stale | Self::Failed)
    }

    pub const fn is_pending(self) -> bool {
        matches!(self, Self::Queued | Self::Rendering)
    }
}

/// Editing a segment that failed to render clears the failure: the next
/// render is a new attempt.
const CLEAR_RENDER_FAILURE: &str = "render_status = CASE WHEN render_status = 'failed' THEN NULL ELSE render_status END, render_error = NULL";

/// The queue position stored on a segment (`render_status`).
const RENDER_STATUS_QUEUED: &str = "queued";
const RENDER_STATUS_RENDERING: &str = "rendering";
const RENDER_STATUS_FAILED: &str = "failed";

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Segments whose audio matches what they would render now.
    pub rendered_segment_count: usize,
    pub stale_segment_count: usize,
    /// Queued or rendering.
    pub pending_segment_count: usize,
    pub failed_segment_count: usize,
    /// Length of the audio the project has, current or stale.
    pub audio_duration_secs: f64,
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
    /// Opaque digest of what the current audio was rendered from.
    #[serde(default)]
    pub render_fingerprint: Option<String>,
    #[serde(default)]
    pub render_state: StudioSegmentRenderState,
    #[serde(default)]
    pub render_error: Option<String>,
    /// The stored queue position; `render_state` is what clients read.
    #[serde(skip)]
    render_status: Option<String>,
}

/// The model, voice and speed a segment renders with: its own override,
/// else the project's.
#[derive(Debug, Clone, PartialEq)]
pub struct StudioRenderSettings {
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
}

/// A segment's own voice override. Model and voice travel together: a
/// speaker only means something for the model it was picked for.
#[derive(Debug, Clone)]
pub struct StudioSegmentVoiceOverride {
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
}

/// A mutation's result plus the Studio audio it no longer references,
/// which the caller deletes.
#[derive(Debug, Clone)]
pub struct StudioProjectChange {
    pub project: StudioProjectRecord,
    pub released_speech_record_ids: Vec<String>,
}

/// One segment the render worker owns until it completes or fails it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudioRenderClaim {
    pub job_id: String,
    pub project_id: String,
    pub segment_id: String,
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
    /// The oldest queued or running render job, if any.
    #[serde(default)]
    pub active_render_job: Option<StudioProjectRenderJobRecord>,
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

    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self { db }
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

        let mut project_rows = rows
            .iter()
            .map(map_project_row)
            .collect::<anyhow::Result<Vec<_>>>()?;

        let has_more = project_rows.len() > page_size;
        if has_more {
            project_rows.truncate(page_size);
        }

        let mut projects = Vec::with_capacity(project_rows.len());
        for mut project in project_rows {
            load_project_segments(db, &mut project).await?;
            projects.push(project_summary(&project));
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
            .begin_with_options(write_transaction_options())
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
            .begin_with_options(write_transaction_options())
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

    /// Change a segment's text, its voice override (`Some(None)` returns it
    /// to the project voice), or both. Its audio stays; the fingerprint
    /// shows it as stale.
    pub async fn update_segment(
        &self,
        project_id: String,
        segment_id: String,
        text: Option<String>,
        voice: Option<Option<StudioSegmentVoiceOverride>>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio segment update transaction")?;
        let now = now_unix_millis_i64();

        if !segment_exists(&tx, &project_id, &segment_id).await? {
            return Ok(None);
        }
        if let Some(text) = text {
            execute(
                &tx,
                format!(
                    "UPDATE studio_project_segments SET text = ?3, updated_at = ?4, {CLEAR_RENDER_FAILURE} WHERE project_id = ?1 AND id = ?2"
                ),
                vec![
                    project_id.clone().into(),
                    segment_id.clone().into(),
                    text.into(),
                    now.into(),
                ],
                "Failed to update Studio segment text",
            )
            .await?;
            sync_project_source_text(&tx, &project_id).await?;
        }
        if let Some(voice) = voice {
            let (model_id, voice_mode, speaker, saved_voice_id) = match voice {
                Some(voice) => (
                    voice.model_id,
                    Some(voice.voice_mode.as_db_value()),
                    voice.speaker,
                    voice.saved_voice_id,
                ),
                None => (None, None, None, None),
            };
            execute(
                &tx,
                format!(
                    r#"
                    UPDATE studio_project_segments
                    SET model_id = ?3, voice_mode = ?4, speaker = ?5, saved_voice_id = ?6,
                        updated_at = ?7, {CLEAR_RENDER_FAILURE}
                    WHERE project_id = ?1 AND id = ?2
                    "#
                ),
                vec![
                    project_id.clone().into(),
                    segment_id.into(),
                    model_id.into(),
                    voice_mode.into(),
                    speaker.into(),
                    saved_voice_id.into(),
                    now.into(),
                ],
                "Failed to update Studio segment voice",
            )
            .await?;
        }

        touch_project(&tx, &project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio segment update transaction")?;
        fetch_project(db, &project_id).await
    }

    /// Add a segment after `after_segment_id`, or at the end. It renders
    /// with the project voice.
    pub async fn insert_segment(
        &self,
        project_id: String,
        after_segment_id: Option<String>,
        text: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio segment insert transaction")?;
        let now = now_unix_millis_i64();

        if !project_exists(&tx, &project_id).await? {
            return Ok(None);
        }
        let insert_position: i64 = if let Some(anchor_segment_id) = after_segment_id {
            let anchor_position = query_one(
                &tx,
                "SELECT position FROM studio_project_segments WHERE project_id = ?1 AND id = ?2",
                vec![project_id.clone().into(), anchor_segment_id.into()],
                "Failed to load Studio segment insert anchor",
            )
            .await?
            .map(|row| row.try_get_by_index::<i64>(0))
            .transpose()?;
            let Some(anchor_position) = anchor_position else {
                return Ok(None);
            };
            anchor_position + 1
        } else {
            query_one(
                &tx,
                "SELECT MAX(position) FROM studio_project_segments WHERE project_id = ?1",
                vec![project_id.clone().into()],
                "Failed to load Studio segment tail position",
            )
            .await?
            .map(|row| row.try_get_by_index::<Option<i64>>(0))
            .transpose()?
            .flatten()
            .map(|value| value + 1)
            .unwrap_or(0)
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
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            vec![
                new_uuid().into(),
                project_id.clone().into(),
                insert_position.into(),
                text.into(),
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
            .begin_with_options(write_transaction_options())
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
                updated_at = ?4,
                render_status = CASE WHEN render_status = 'failed' THEN NULL ELSE render_status END,
                render_error = NULL
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
    ) -> anyhow::Result<Option<StudioProjectChange>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio segment delete transaction")?;
        let now = now_unix_millis_i64();

        let segment = query_one(
            &tx,
            r#"
            SELECT position, speech_record_id
            FROM studio_project_segments
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![project_id.clone().into(), segment_id.clone().into()],
            "Failed to load Studio segment position for delete",
        )
        .await?
        .map(|row| {
            Ok::<_, anyhow::Error>((
                row.try_get_by_index::<i64>(0)?,
                row.try_get_by_index::<Option<String>>(1)?,
            ))
        })
        .transpose()?;

        let Some((position, speech_record_id)) = segment else {
            return Ok(None);
        };
        let remaining = query_one(
            &tx,
            "SELECT COUNT(*) FROM studio_project_segments WHERE project_id = ?1",
            vec![project_id.clone().into()],
            "Failed to count Studio segments for delete",
        )
        .await?
        .map(|row| row.try_get_by_index::<i64>(0))
        .transpose()?
        .unwrap_or_default();
        if remaining <= 1 {
            return Err(StoreError::invalid("A project must keep at least one segment.").into());
        }

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
        project_change(db, &project_id, speech_record_id.into_iter().collect()).await
    }

    pub async fn merge_segment_with_next(
        &self,
        project_id: String,
        segment_id: String,
    ) -> anyhow::Result<Option<StudioProjectChange>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
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
            SELECT id, position, text, speech_record_id
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
                row.try_get_by_index::<Option<String>>(3)?,
            ))
        })
        .transpose()?;
        let Some((next_id, next_position, next_text, next_record_id)) = next else {
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
                updated_at = ?4,
                render_status = CASE WHEN render_status = 'failed' THEN NULL ELSE render_status END,
                render_error = NULL
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
        project_change(db, &project_id, next_record_id.into_iter().collect()).await
    }

    pub async fn reorder_segments(
        &self,
        project_id: String,
        ordered_segment_ids: Vec<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
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
            return Err(StoreError::invalid(
                "Reorder request must include every project segment exactly once.",
            )
            .into());
        }

        let existing_set = existing_ids.iter().collect::<HashSet<_>>();
        let requested_set = ordered_segment_ids.iter().collect::<HashSet<_>>();
        if existing_set != requested_set {
            return Err(StoreError::invalid(
                "Reorder request contains unknown or missing segment ids.",
            )
            .into());
        }

        let now = now_unix_millis_i64();
        for (position, segment_id) in ordered_segment_ids.iter().enumerate() {
            execute(
                &tx,
                r#"
                UPDATE studio_project_segments
                SET position = ?3, updated_at = ?4
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
    ) -> anyhow::Result<Option<StudioProjectChange>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio segment batch delete transaction")?;
        let rows = query_all(
            &tx,
            r#"
            SELECT id, speech_record_id
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
                    row.try_get_by_index::<Option<String>>(1)?,
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
            return project_change(db, &project_id, Vec::new()).await;
        }
        let released = existing
            .iter()
            .filter(|(id, _)| remove_set.contains(id))
            .filter_map(|(_, record_id)| record_id.clone())
            .collect::<Vec<_>>();

        let remaining_count = existing
            .iter()
            .filter(|(id, _)| !remove_set.contains(id))
            .count();
        if remaining_count == 0 {
            return Err(StoreError::invalid("A project must keep at least one segment.").into());
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
        project_change(db, &project_id, released).await
    }

    /// Delete a project. Returns the Studio audio it referenced, or `None`
    /// when there was no such project.
    pub async fn delete_project(&self, project_id: String) -> anyhow::Result<Option<Vec<String>>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio project delete transaction")?;
        let released = query_all(
            &tx,
            "SELECT speech_record_id FROM studio_project_segments WHERE project_id = ?1 AND speech_record_id IS NOT NULL",
            vec![project_id.clone().into()],
            "Failed to list Studio project audio for delete",
        )
        .await?
        .iter()
        .map(|row| row.try_get_by_index::<String>(0).map_err(Into::into))
        .collect::<anyhow::Result<Vec<_>>>()?;
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
        Ok((deleted > 0).then_some(released))
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
            .begin_with_options(write_transaction_options())
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

        if current.is_some() {
            execute(
                &tx,
                r#"
                UPDATE studio_project_meta
                SET
                    folder_id = ?2,
                    tags_json = ?3,
                    default_export_format = ?4,
                    last_render_job_id = ?5,
                    last_rendered_at = ?6
                WHERE project_id = ?1
                "#,
                vec![
                    project_id.clone().into(),
                    next_folder_id.into(),
                    encode_json_string_vec(next_tags.as_slice()).into(),
                    next_export.as_db_value().into(),
                    next_last_render_job_id.into(),
                    next_last_rendered_at_i64.into(),
                ],
                "Failed to update Studio project metadata",
            )
            .await?;
        } else {
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
                "#,
                vec![
                    project_id.clone().into(),
                    next_folder_id.into(),
                    encode_json_string_vec(next_tags.as_slice()).into(),
                    next_export.as_db_value().into(),
                    next_last_render_job_id.into(),
                    next_last_rendered_at_i64.into(),
                ],
                "Failed to insert Studio project metadata",
            )
            .await?;
        }

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
        fetch_pronunciations(db, &project_id).await
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
            studio_project_snapshots_sql(db)?,
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
            .begin_with_options(write_transaction_options())
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
            .begin_with_options(write_transaction_options())
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

    /// Queue segments for the render worker: `segment_ids`, or every
    /// segment whose audio is missing, stale or failed. Segments already
    /// queued or rendering are left where they are.
    pub async fn enqueue_render(
        &self,
        project_id: String,
        segment_ids: Option<Vec<String>>,
    ) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio render enqueue transaction")?;
        let Some(project) = fetch_project(&tx, &project_id).await? else {
            return Ok(None);
        };

        let queued = match segment_ids {
            Some(requested) => {
                let requested = requested.into_iter().collect::<HashSet<_>>();
                let known = project
                    .segments
                    .iter()
                    .filter(|segment| requested.contains(&segment.id))
                    .count();
                if known != requested.len() {
                    return Err(StoreError::invalid(
                        "Render requests can only name segments of this project.",
                    )
                    .into());
                }
                project
                    .segments
                    .iter()
                    .filter(|segment| requested.contains(&segment.id))
                    .filter(|segment| !segment.render_state.is_pending())
                    .map(|segment| segment.id.clone())
                    .collect::<Vec<_>>()
            }
            None => project
                .segments
                .iter()
                .filter(|segment| segment.render_state.needs_render())
                .map(|segment| segment.id.clone())
                .collect::<Vec<_>>(),
        };
        if queued.is_empty() {
            return Err(StoreError::invalid(
                "Nothing to render: those paragraphs are current or already queued.",
            )
            .into());
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
                encode_json_string_vec(queued.as_slice()).into(),
            ],
            "Failed to create Studio project render job",
        )
        .await?;
        for segment_id in &queued {
            execute(
                &tx,
                "UPDATE studio_project_segments SET render_status = ?3, render_error = NULL WHERE project_id = ?1 AND id = ?2",
                vec![
                    project_id.clone().into(),
                    segment_id.clone().into(),
                    RENDER_STATUS_QUEUED.into(),
                ],
                "Failed to queue Studio segment",
            )
            .await?;
        }
        upsert_last_render_job(&tx, &project_id, &job_id, None).await?;

        tx.commit()
            .await
            .context("Failed to commit Studio render enqueue transaction")?;
        fetch_render_job(db, &project_id, &job_id).await
    }

    /// Hand the oldest job's next queued segment to the render worker, and
    /// finish jobs that have nothing left.
    pub async fn claim_next_render(&self) -> anyhow::Result<Option<StudioRenderClaim>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio render claim transaction")?;
        let jobs = query_all(
            &tx,
            r#"
            SELECT id, project_id, queued_segment_ids_json
            FROM studio_project_render_jobs
            WHERE status IN ('queued', 'running')
            ORDER BY created_at ASC, id ASC
            "#,
            vec![],
            "Failed to list active Studio render jobs",
        )
        .await?;

        let now = now_unix_millis_i64();
        for row in jobs {
            let job_id: String = row.try_get_by_index(0)?;
            let project_id: String = row.try_get_by_index(1)?;
            let job_segment_ids = parse_json_string_vec(row.try_get_by_index(2)?);
            let statuses = segment_render_statuses(&tx, &project_id).await?;
            let status_of = |id: &String| statuses.get(id).and_then(|status| status.as_deref());

            if let Some(segment_id) = job_segment_ids
                .iter()
                .find(|id| status_of(id) == Some(RENDER_STATUS_QUEUED))
            {
                execute(
                    &tx,
                    "UPDATE studio_project_segments SET render_status = ?3 WHERE project_id = ?1 AND id = ?2",
                    vec![
                        project_id.clone().into(),
                        segment_id.clone().into(),
                        RENDER_STATUS_RENDERING.into(),
                    ],
                    "Failed to start Studio segment render",
                )
                .await?;
                execute(
                    &tx,
                    "UPDATE studio_project_render_jobs SET status = ?2, updated_at = ?3 WHERE id = ?1",
                    vec![
                        job_id.clone().into(),
                        StudioProjectRenderJobStatus::Running.as_db_value().into(),
                        now.into(),
                    ],
                    "Failed to start Studio render job",
                )
                .await?;
                tx.commit()
                    .await
                    .context("Failed to commit Studio render claim transaction")?;
                return Ok(Some(StudioRenderClaim {
                    job_id,
                    project_id,
                    segment_id: segment_id.clone(),
                }));
            }
            if job_segment_ids
                .iter()
                .any(|id| status_of(id) == Some(RENDER_STATUS_RENDERING))
            {
                continue;
            }
            finish_render_job(&tx, &project_id, &job_id, &job_segment_ids, now).await?;
        }

        tx.commit()
            .await
            .context("Failed to commit Studio render claim transaction")?;
        Ok(None)
    }

    /// Attach a finished render. Returns the Studio audio nothing references
    /// any more: the segment's previous take, or this one when the segment
    /// was deleted while it rendered.
    pub async fn complete_render(
        &self,
        claim: &StudioRenderClaim,
        speech_record_id: String,
        render_fingerprint: String,
    ) -> anyhow::Result<Vec<String>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio render completion transaction")?;
        let previous = query_one(
            &tx,
            "SELECT speech_record_id FROM studio_project_segments WHERE project_id = ?1 AND id = ?2",
            vec![
                claim.project_id.clone().into(),
                claim.segment_id.clone().into(),
            ],
            "Failed to load Studio segment for render completion",
        )
        .await?
        .map(|row| row.try_get_by_index::<Option<String>>(0))
        .transpose()?;
        let Some(previous) = previous else {
            tx.commit()
                .await
                .context("Failed to commit Studio render completion transaction")?;
            return Ok(vec![speech_record_id]);
        };

        let now = now_unix_millis_i64();
        execute(
            &tx,
            r#"
            UPDATE studio_project_segments
            SET
                speech_record_id = ?3,
                render_fingerprint = ?4,
                render_status = NULL,
                render_error = NULL,
                updated_at = ?5
            WHERE project_id = ?1 AND id = ?2
            "#,
            vec![
                claim.project_id.clone().into(),
                claim.segment_id.clone().into(),
                speech_record_id.clone().into(),
                render_fingerprint.into(),
                now.into(),
            ],
            "Failed to attach Studio segment render",
        )
        .await?;
        touch_project(&tx, &claim.project_id, now).await?;
        tx.commit()
            .await
            .context("Failed to commit Studio render completion transaction")?;
        Ok(previous
            .filter(|previous| *previous != speech_record_id)
            .into_iter()
            .collect())
    }

    /// Record why a claimed segment did not render.
    pub async fn fail_render(&self, claim: &StudioRenderClaim, error: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        execute(
            db,
            "UPDATE studio_project_segments SET render_status = ?3, render_error = ?4 WHERE project_id = ?1 AND id = ?2 AND render_status = ?5",
            vec![
                claim.project_id.clone().into(),
                claim.segment_id.clone().into(),
                RENDER_STATUS_FAILED.into(),
                error.into(),
                RENDER_STATUS_RENDERING.into(),
            ],
            "Failed to record Studio segment render failure",
        )
        .await?;
        Ok(())
    }

    /// Fail every segment still queued in a job, for errors no retry inside
    /// the job can fix (the model isn't on this device).
    pub async fn fail_queued_in_job(
        &self,
        claim: &StudioRenderClaim,
        error: &str,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        let Some(job) = fetch_render_job(db, &claim.project_id, &claim.job_id).await? else {
            return Ok(());
        };
        for segment_id in job.queued_segment_ids {
            execute(
                db,
                "UPDATE studio_project_segments SET render_status = ?3, render_error = ?4 WHERE project_id = ?1 AND id = ?2 AND render_status = ?5",
                vec![
                    claim.project_id.clone().into(),
                    segment_id.into(),
                    RENDER_STATUS_FAILED.into(),
                    error.into(),
                    RENDER_STATUS_QUEUED.into(),
                ],
                "Failed to record Studio job render failure",
            )
            .await?;
        }
        Ok(())
    }

    /// Stop a job: its queued segments leave the queue; a segment already
    /// rendering finishes.
    pub async fn cancel_render_job(
        &self,
        project_id: String,
        job_id: String,
    ) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio render cancel transaction")?;
        let Some(job) = fetch_render_job(&tx, &project_id, &job_id).await? else {
            return Ok(None);
        };
        if matches!(
            job.status,
            StudioProjectRenderJobStatus::Queued | StudioProjectRenderJobStatus::Running
        ) {
            for segment_id in &job.queued_segment_ids {
                execute(
                    &tx,
                    "UPDATE studio_project_segments SET render_status = NULL WHERE project_id = ?1 AND id = ?2 AND render_status = ?3",
                    vec![
                        project_id.clone().into(),
                        segment_id.clone().into(),
                        RENDER_STATUS_QUEUED.into(),
                    ],
                    "Failed to unqueue Studio segment",
                )
                .await?;
            }
            execute(
                &tx,
                "UPDATE studio_project_render_jobs SET status = ?3, updated_at = ?4 WHERE project_id = ?1 AND id = ?2",
                vec![
                    project_id.clone().into(),
                    job_id.clone().into(),
                    StudioProjectRenderJobStatus::Cancelled.as_db_value().into(),
                    now_unix_millis_i64().into(),
                ],
                "Failed to cancel Studio render job",
            )
            .await?;
        }
        tx.commit()
            .await
            .context("Failed to commit Studio render cancel transaction")?;
        fetch_render_job(db, &project_id, &job_id).await
    }

    /// A render interrupted by a restart goes back to the queue, if its
    /// job is still active; otherwise (its job was cancelled meanwhile) it
    /// leaves the queue.
    pub async fn requeue_interrupted_renders(&self) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start Studio render recovery transaction")?;
        let requeued = execute(
            &tx,
            "UPDATE studio_project_segments SET render_status = ?1 WHERE render_status = ?2",
            vec![RENDER_STATUS_QUEUED.into(), RENDER_STATUS_RENDERING.into()],
            "Failed to requeue interrupted Studio renders",
        )
        .await?;

        let mut in_active_jobs = HashSet::new();
        for row in query_all(
            &tx,
            "SELECT queued_segment_ids_json FROM studio_project_render_jobs WHERE status IN ('queued', 'running')",
            vec![],
            "Failed to list active Studio render jobs",
        )
        .await?
        {
            in_active_jobs.extend(parse_json_string_vec(row.try_get_by_index(0)?));
        }
        let queued = query_all(
            &tx,
            "SELECT id FROM studio_project_segments WHERE render_status = ?1",
            vec![RENDER_STATUS_QUEUED.into()],
            "Failed to list queued Studio segments",
        )
        .await?
        .iter()
        .map(|row| row.try_get_by_index::<String>(0).map_err(Into::into))
        .collect::<anyhow::Result<Vec<_>>>()?;
        for segment_id in queued.into_iter().filter(|id| !in_active_jobs.contains(id)) {
            execute(
                &tx,
                "UPDATE studio_project_segments SET render_status = NULL WHERE id = ?1",
                vec![segment_id.into()],
                "Failed to unqueue orphaned Studio segment",
            )
            .await?;
        }
        tx.commit()
            .await
            .context("Failed to commit Studio render recovery transaction")?;
        Ok(requeued)
    }

    /// Record what audio rendered before fingerprints existed was rendered
    /// from, taking the project's current settings as the answer. Runs once
    /// per database in practice: every render stamps its own.
    pub async fn stamp_unfingerprinted_renders(&self) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let project_ids = query_all(
            db,
            "SELECT DISTINCT project_id FROM studio_project_segments WHERE speech_record_id IS NOT NULL AND render_fingerprint IS NULL",
            vec![],
            "Failed to list Studio projects without render fingerprints",
        )
        .await?
        .iter()
        .map(|row| row.try_get_by_index::<String>(0).map_err(Into::into))
        .collect::<anyhow::Result<Vec<_>>>()?;

        let mut stamped = 0;
        for project_id in project_ids {
            let tx = db
                .begin_with_options(write_transaction_options())
                .await
                .context("Failed to start Studio fingerprint transaction")?;
            let Some(project) = fetch_project(&tx, &project_id).await? else {
                continue;
            };
            let pronunciations = fetch_pronunciations(&tx, &project_id).await?;
            for segment in project.segments.iter().filter(|segment| {
                segment.speech_record_id.is_some() && segment.render_fingerprint.is_none()
            }) {
                let fingerprint = segment_render_fingerprint(&project, segment, &pronunciations);
                stamped += execute(
                    &tx,
                    "UPDATE studio_project_segments SET render_fingerprint = ?3 WHERE project_id = ?1 AND id = ?2",
                    vec![
                        project_id.clone().into(),
                        segment.id.clone().into(),
                        fingerprint.into(),
                    ],
                    "Failed to stamp Studio render fingerprint",
                )
                .await?;
            }
            tx.commit()
                .await
                .context("Failed to commit Studio fingerprint transaction")?;
        }
        Ok(stamped)
    }
}

/// The settings a segment renders with: its override, else the project's.
pub fn resolve_segment_render_settings(
    project: &StudioProjectRecord,
    segment: &StudioProjectSegmentRecord,
) -> StudioRenderSettings {
    let voice_mode = segment.voice_mode.unwrap_or(project.voice_mode);
    let speaker = match voice_mode {
        StudioProjectVoiceMode::BuiltIn => {
            segment.speaker.clone().or_else(|| project.speaker.clone())
        }
        StudioProjectVoiceMode::Saved => None,
    };
    let saved_voice_id = match voice_mode {
        StudioProjectVoiceMode::BuiltIn => None,
        StudioProjectVoiceMode::Saved => segment
            .saved_voice_id
            .clone()
            .or_else(|| project.saved_voice_id.clone()),
    };
    StudioRenderSettings {
        model_id: segment
            .model_id
            .clone()
            .or_else(|| project.model_id.clone()),
        voice_mode,
        speaker,
        saved_voice_id,
        speed: project.speed,
    }
}

/// Apply pronunciation rules in one pass: whole words, any letter case,
/// longest rule first, and a replacement is never matched again.
pub fn apply_pronunciations(text: &str, rules: &[StudioProjectPronunciationRecord]) -> String {
    let mut rules = rules
        .iter()
        .filter(|rule| !rule.source_text.trim().is_empty())
        .map(|rule| {
            (
                rule.source_text.trim().chars().collect::<Vec<_>>(),
                rule.replacement_text.as_str(),
            )
        })
        .collect::<Vec<_>>();
    if rules.is_empty() {
        return text.to_string();
    }
    rules.sort_by_key(|rule| std::cmp::Reverse(rule.0.len()));

    let chars = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    'scan: while index < chars.len() {
        for (source, replacement) in &rules {
            if matches_whole_word(&chars, index, source) {
                output.push_str(replacement);
                index += source.len();
                continue 'scan;
            }
        }
        output.push(chars[index]);
        index += 1;
    }
    output
}

fn matches_whole_word(text: &[char], start: usize, word: &[char]) -> bool {
    let end = start + word.len();
    if end > text.len() {
        return false;
    }
    let same = text[start..end]
        .iter()
        .zip(word)
        .all(|(left, right)| left.to_lowercase().eq(right.to_lowercase()));
    if !same {
        return false;
    }
    // A boundary only matters next to a word character of the rule itself,
    // so rules such as "C++" or "e.g." still match.
    let starts_clean =
        !word[0].is_alphanumeric() || start == 0 || !text[start - 1].is_alphanumeric();
    let ends_clean = !word[word.len() - 1].is_alphanumeric()
        || end == text.len()
        || !text[end].is_alphanumeric();
    starts_clean && ends_clean
}

/// What a render of this text with these settings depends on.
pub fn render_fingerprint(text: &str, settings: &StudioRenderSettings) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for part in [
        "studio-render-v1",
        settings.model_id.as_deref().unwrap_or_default(),
        settings.voice_mode.as_db_value(),
        settings.speaker.as_deref().unwrap_or_default(),
        settings.saved_voice_id.as_deref().unwrap_or_default(),
        &format!("{:.3}", settings.speed.unwrap_or(1.0)),
        text,
    ] {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    hasher
        .finalize()
        .iter()
        .take(16)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The text a segment sends to the model and the fingerprint of rendering
/// it now.
pub fn segment_render_input(
    project: &StudioProjectRecord,
    segment: &StudioProjectSegmentRecord,
    pronunciations: &[StudioProjectPronunciationRecord],
) -> (String, StudioRenderSettings, String) {
    let settings = resolve_segment_render_settings(project, segment);
    let text = apply_pronunciations(&segment.text, pronunciations);
    let fingerprint = render_fingerprint(&text, &settings);
    (text, settings, fingerprint)
}

fn segment_render_fingerprint(
    project: &StudioProjectRecord,
    segment: &StudioProjectSegmentRecord,
    pronunciations: &[StudioProjectPronunciationRecord],
) -> String {
    segment_render_input(project, segment, pronunciations).2
}

// The list reads only the columns a summary needs: no source text.
const STUDIO_PROJECT_PAGE_SQL: &str = r#"
    SELECT
        id,
        created_at,
        updated_at,
        name,
        source_filename,
        '' AS source_text,
        model_id,
        voice_mode,
        speaker,
        saved_voice_id,
        speed
    FROM studio_projects
    ORDER BY updated_at DESC, id DESC
    LIMIT ?1
"#;

const STUDIO_PROJECT_PAGE_AFTER_CURSOR_SQL: &str = r#"
    SELECT
        id,
        created_at,
        updated_at,
        name,
        source_filename,
        '' AS source_text,
        model_id,
        voice_mode,
        speaker,
        saved_voice_id,
        speed
    FROM studio_projects
    WHERE updated_at < ?1 OR (updated_at = ?1 AND id < ?2)
    ORDER BY updated_at DESC, id DESC
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

// Segment audio is a Studio speech record; projects rendered before Studio
// had its own route kind point at text-to-speech records.
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
        CASE WHEN h.id IS NULL THEN NULL ELSE s.speech_record_id END AS speech_record_id,
        s.updated_at,
        h.generation_time_ms,
        h.audio_duration_secs,
        h.audio_filename,
        s.render_fingerprint,
        s.render_status,
        s.render_error
    FROM studio_project_segments s
    LEFT JOIN speech_history_records h
        ON h.id = s.speech_record_id
       AND h.route_kind IN ('studio', 'text_to_speech')
    WHERE s.project_id = ?1
    ORDER BY s.position ASC, s.id ASC
"#;

const STUDIO_PROJECT_PRONUNCIATIONS_SQL: &str = r#"
    SELECT id, project_id, source_text, replacement_text, locale, created_at, updated_at
    FROM studio_project_pronunciations
    WHERE project_id = ?1
    ORDER BY updated_at DESC, id DESC
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

fn studio_project_snapshots_sql<C>(db: &C) -> anyhow::Result<String>
where
    C: ConnectionTrait,
{
    studio_project_snapshot_summary_sql(
        db,
        "WHERE s.project_id = ?1\n    ORDER BY s.created_at DESC, s.id DESC",
    )
}

fn studio_project_snapshot_by_id_sql<C>(db: &C) -> anyhow::Result<String>
where
    C: ConnectionTrait,
{
    studio_project_snapshot_summary_sql(db, "WHERE s.id = ?1")
}

fn studio_project_snapshot_summary_sql<C>(db: &C, predicate: &str) -> anyhow::Result<String>
where
    C: ConnectionTrait,
{
    let project_name = raw::json_extract_text(db.get_database_backend(), "s.project_json", "name")?;
    Ok(format!(
        r#"
    SELECT
        s.id,
        s.project_id,
        s.created_at,
        s.label,
        COALESCE({project_name}, p.name) AS project_name
    FROM studio_project_snapshots s
    LEFT JOIN studio_projects p ON p.id = s.project_id
    {predicate}
"#
    ))
}

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

const STUDIO_PROJECT_ACTIVE_RENDER_JOB_SQL: &str = r#"
    SELECT
        id,
        project_id,
        created_at,
        updated_at,
        status,
        error_message,
        queued_segment_ids_json
    FROM studio_project_render_jobs
    WHERE project_id = ?1 AND status IN ('queued', 'running')
    ORDER BY created_at ASC, id ASC
    LIMIT 1
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
    load_project_segments(db, &mut project).await?;
    project.active_render_job = query_one(
        db,
        STUDIO_PROJECT_ACTIVE_RENDER_JOB_SQL,
        vec![project_id.into()],
        "Failed to load active Studio render job",
    )
    .await?
    .as_ref()
    .map(map_render_job_row)
    .transpose()?;
    Ok(Some(project))
}

async fn project_change<C>(
    db: &C,
    project_id: &str,
    released_speech_record_ids: Vec<String>,
) -> anyhow::Result<Option<StudioProjectChange>>
where
    C: ConnectionTrait,
{
    Ok(fetch_project(db, project_id)
        .await?
        .map(|project| StudioProjectChange {
            project,
            released_speech_record_ids,
        }))
}

async fn segment_exists<C>(db: &C, project_id: &str, segment_id: &str) -> anyhow::Result<bool>
where
    C: ConnectionTrait,
{
    Ok(query_one(
        db,
        "SELECT 1 FROM studio_project_segments WHERE project_id = ?1 AND id = ?2",
        vec![project_id.into(), segment_id.into()],
        "Failed to check Studio segment existence",
    )
    .await?
    .is_some())
}

async fn segment_render_statuses<C>(
    db: &C,
    project_id: &str,
) -> anyhow::Result<HashMap<String, Option<String>>>
where
    C: ConnectionTrait,
{
    query_all(
        db,
        "SELECT id, render_status FROM studio_project_segments WHERE project_id = ?1",
        vec![project_id.into()],
        "Failed to load Studio segment render statuses",
    )
    .await?
    .iter()
    .map(|row| {
        Ok((
            row.try_get_by_index::<String>(0)?,
            row.try_get_by_index::<Option<String>>(1)?,
        ))
    })
    .collect()
}

/// Close a job whose segments have all left the queue: failed when any of
/// them failed, with the first failure as its error.
async fn finish_render_job<C>(
    db: &C,
    project_id: &str,
    job_id: &str,
    job_segment_ids: &[String],
    now: i64,
) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    let mut failures = Vec::new();
    for segment_id in job_segment_ids {
        let failure = query_one(
            db,
            "SELECT render_error FROM studio_project_segments WHERE project_id = ?1 AND id = ?2 AND render_status = ?3",
            vec![
                project_id.into(),
                segment_id.as_str().into(),
                RENDER_STATUS_FAILED.into(),
            ],
            "Failed to load Studio segment render failure",
        )
        .await?
        .map(|row| row.try_get_by_index::<Option<String>>(0))
        .transpose()?;
        if let Some(error) = failure {
            failures.push(error.unwrap_or_default());
        }
    }
    let (status, error_message) = match failures.first() {
        None => (StudioProjectRenderJobStatus::Completed, None),
        Some(first) => (
            StudioProjectRenderJobStatus::Failed,
            Some(match failures.len() {
                1 => format!("1 paragraph failed: {first}"),
                count => format!("{count} paragraphs failed: {first}"),
            }),
        ),
    };
    execute(
        db,
        "UPDATE studio_project_render_jobs SET status = ?2, error_message = ?3, updated_at = ?4 WHERE id = ?1",
        vec![
            job_id.into(),
            status.as_db_value().into(),
            error_message.into(),
            now.into(),
        ],
        "Failed to finish Studio render job",
    )
    .await?;
    upsert_last_render_job(db, project_id, job_id, Some(now)).await
}

/// Load a project's segments and work out where each one's audio stands.
async fn load_project_segments<C>(db: &C, project: &mut StudioProjectRecord) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    let rows = query_all(
        db,
        STUDIO_PROJECT_SEGMENTS_SQL,
        vec![project.id.as_str().into()],
        "Failed to load Studio project segments",
    )
    .await?;
    let segments = rows
        .iter()
        .map(map_segment_row)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let pronunciations = fetch_pronunciations(db, &project.id).await?;
    project.segments = segments
        .into_iter()
        .map(|mut segment| {
            segment.render_state = segment_render_state(project, &segment, &pronunciations);
            if segment.render_state != StudioSegmentRenderState::Failed {
                segment.render_error = None;
            }
            segment
        })
        .collect();
    Ok(())
}

async fn fetch_pronunciations<C>(
    db: &C,
    project_id: &str,
) -> anyhow::Result<Vec<StudioProjectPronunciationRecord>>
where
    C: ConnectionTrait,
{
    let rows = query_all(
        db,
        STUDIO_PROJECT_PRONUNCIATIONS_SQL,
        vec![project_id.into()],
        "Failed to list Studio project pronunciations",
    )
    .await?;
    rows.iter().map(map_pronunciation_row).collect()
}

fn segment_render_state(
    project: &StudioProjectRecord,
    segment: &StudioProjectSegmentRecord,
    pronunciations: &[StudioProjectPronunciationRecord],
) -> StudioSegmentRenderState {
    match segment.render_status.as_deref() {
        Some(RENDER_STATUS_QUEUED) => return StudioSegmentRenderState::Queued,
        Some(RENDER_STATUS_RENDERING) => return StudioSegmentRenderState::Rendering,
        Some(RENDER_STATUS_FAILED) => return StudioSegmentRenderState::Failed,
        _ => {}
    }
    if segment.speech_record_id.is_none() {
        return StudioSegmentRenderState::NotRendered;
    }
    // Audio rendered before fingerprints existed is taken as current; the
    // render worker stamps it once at start-up.
    let Some(rendered_from) = segment.render_fingerprint.as_deref() else {
        return StudioSegmentRenderState::Rendered;
    };
    if rendered_from == segment_render_fingerprint(project, segment, pronunciations) {
        StudioSegmentRenderState::Rendered
    } else {
        StudioSegmentRenderState::Stale
    }
}

fn project_summary(project: &StudioProjectRecord) -> StudioProjectSummary {
    let count = |state: StudioSegmentRenderState| {
        project
            .segments
            .iter()
            .filter(|segment| segment.render_state == state)
            .count()
    };
    StudioProjectSummary {
        id: project.id.clone(),
        created_at: project.created_at,
        updated_at: project.updated_at,
        name: project.name.clone(),
        source_filename: project.source_filename.clone(),
        model_id: project.model_id.clone(),
        voice_mode: project.voice_mode,
        speaker: project.speaker.clone(),
        saved_voice_id: project.saved_voice_id.clone(),
        speed: project.speed,
        segment_count: project.segments.len(),
        rendered_segment_count: count(StudioSegmentRenderState::Rendered),
        stale_segment_count: count(StudioSegmentRenderState::Stale),
        pending_segment_count: project
            .segments
            .iter()
            .filter(|segment| segment.render_state.is_pending())
            .count(),
        failed_segment_count: count(StudioSegmentRenderState::Failed),
        // `sum` of no floats is -0.0; start from 0.0.
        audio_duration_secs: project
            .segments
            .iter()
            .filter_map(|segment| segment.audio_duration_secs)
            .fold(0.0, |total, secs| total + secs),
        total_chars: project
            .segments
            .iter()
            .map(|segment| segment.input_chars)
            .sum(),
    }
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
        studio_project_snapshot_by_id_sql(db)?,
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
    let exists = query_one(
        db,
        "SELECT 1 FROM studio_project_meta WHERE project_id = ?1",
        vec![project_id.into()],
        "Failed to check Studio project render metadata",
    )
    .await?
    .is_some();

    match (exists, last_rendered_at) {
        (true, Some(last_rendered_at)) => {
            execute(
                db,
                r#"
                UPDATE studio_project_meta
                SET
                    last_render_job_id = ?2,
                    last_rendered_at = ?3
                WHERE project_id = ?1
                "#,
                vec![project_id.into(), job_id.into(), last_rendered_at.into()],
                "Failed to update Studio project completed render metadata",
            )
            .await?;
        }
        (true, None) => {
            execute(
                db,
                r#"
                UPDATE studio_project_meta
                SET last_render_job_id = ?2
                WHERE project_id = ?1
                "#,
                vec![project_id.into(), job_id.into()],
                "Failed to update Studio project render metadata",
            )
            .await?;
        }
        (false, Some(last_rendered_at)) => {
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
                VALUES (?1, NULL, '[]', 'wav', ?2, ?3)
                "#,
                vec![project_id.into(), job_id.into(), last_rendered_at.into()],
                "Failed to insert Studio project completed render metadata",
            )
            .await?;
        }
        (false, None) => {
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
                VALUES (?1, NULL, '[]', 'wav', ?2, NULL)
                "#,
                vec![project_id.into(), job_id.into()],
                "Failed to insert Studio project render metadata",
            )
            .await?;
        }
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
    let statement = raw::statement(db, sql, values).context(context)?;
    let result = db.execute_raw(statement).await.context(context)?;
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
    let statement = raw::statement(db, sql, values).context(context)?;
    db.query_one_raw(statement).await.context(context)
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
    let statement = raw::statement(db, sql, values).context(context)?;
    db.query_all_raw(statement).await.context(context)
}

/// A stored model id as the application reads it: a removed variant is
/// replaced by its supported counterpart, so projects saved before the
/// removal keep rendering (P1.0).
fn normalize_stored_model_id(model_id: Option<String>) -> Option<String> {
    model_id.map(|id| izwi_core::normalize_stored_model_id(&id).into_owned())
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
        model_id: normalize_stored_model_id(row.try_get_by_index(6)?),
        voice_mode: StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(StudioProjectVoiceMode::BuiltIn),
        speaker: row.try_get_by_index(8)?,
        saved_voice_id: row.try_get_by_index(9)?,
        speed: row.try_get_by_index(10)?,
        segments: Vec::new(),
        active_render_job: None,
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
        model_id: normalize_stored_model_id(row.try_get_by_index(4)?),
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
        render_fingerprint: row.try_get_by_index(13)?,
        render_status: row.try_get_by_index(14)?,
        render_error: row.try_get_by_index(15)?,
        render_state: StudioSegmentRenderState::NotRendered,
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
        apply_pronunciations, segment_render_input, NewStudioProjectPronunciationRecord,
        NewStudioProjectRecord, NewStudioProjectSegment, StudioProjectPronunciationRecord,
        StudioProjectRecord, StudioProjectRenderJobStatus, StudioProjectStore,
        StudioProjectVoiceMode, StudioRenderClaim, StudioSegmentRenderState,
        StudioSegmentVoiceOverride, UpdateStudioProjectRecord,
    };
    use sea_orm::ConnectionTrait;
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

    /// P1.0: a project saved with a removed model id reads back with the
    /// supported counterpart, so it keeps rendering after the removal.
    #[tokio::test]
    async fn stored_removed_model_ids_read_back_as_their_dense_replacement() {
        let root = test_env_root("studio-project-removed-model");
        std::fs::create_dir_all(&root).expect("root should be created");
        let store = initialize_test_store(root.join("test.sqlite3"), root.join("media")).await;
        let removed = "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit".to_string();
        let created = store
            .create_project(NewStudioProjectRecord {
                name: "Old project".to_string(),
                source_filename: None,
                source_text: "Hello world.".to_string(),
                model_id: Some(removed.clone()),
                voice_mode: StudioProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: None,
                segments: vec![NewStudioProjectSegment {
                    position: 0,
                    text: "Hello world.".to_string(),
                    model_id: Some(removed),
                    voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                    speaker: Some("Vivian".to_string()),
                    saved_voice_id: None,
                }],
            })
            .await
            .expect("project should be created");
        let dense = Some("Qwen3-TTS-12Hz-0.6B-CustomVoice");
        assert_eq!(created.model_id.as_deref(), dense);
        assert_eq!(created.segments[0].model_id.as_deref(), dense);
        let read = store
            .get_project(created.id.clone())
            .await
            .expect("read should succeed")
            .expect("project should exist");
        assert_eq!(read.model_id.as_deref(), dense);
        assert_eq!(read.segments[0].model_id.as_deref(), dense);
        let _ = std::fs::remove_dir_all(&root);
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
            .update_segment(
                updated.id.clone(),
                segment.id.clone(),
                Some("Updated line.".to_string()),
                None,
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
            .expect("project should exist")
            .project;

        assert_eq!(deleted.segments.len(), 2);
        assert_eq!(deleted.segments[0].position, 0);
        assert_eq!(deleted.segments[1].position, 1);
        assert_eq!(deleted.segments[0].text, "Hello world.");
        assert_eq!(deleted.segments[1].text, "Closing line.");
        assert_eq!(deleted.source_text, "Hello world.\n\nClosing line.");

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn insert_segment_places_block_after_anchor_with_the_project_voice() {
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
        // The anchor's own voice stays with the anchor; a new paragraph
        // follows the project, so project voice changes reach it.
        assert_eq!(inserted.segments[1].model_id, None);
        assert_eq!(inserted.segments[1].voice_mode, None);
        assert_eq!(inserted.segments[1].speaker, None);
        assert_eq!(inserted.segments[1].saved_voice_id, None);
        assert_eq!(
            inserted.source_text,
            "Intro block.\n\nMiddle block.\n\nClosing block."
        );

        let _ = std::fs::remove_dir_all(root);
    }

    fn narration(name: &str) -> NewStudioProjectRecord {
        let segment = |position, text: &str| NewStudioProjectSegment {
            position,
            text: text.to_string(),
            model_id: None,
            voice_mode: None,
            speaker: None,
            saved_voice_id: None,
        };
        NewStudioProjectRecord {
            name: name.to_string(),
            source_filename: None,
            source_text: "The harbour woke. Mara walked the quay.".to_string(),
            model_id: Some("Kokoro-82M".to_string()),
            voice_mode: StudioProjectVoiceMode::BuiltIn,
            speaker: Some("af_heart".to_string()),
            saved_voice_id: None,
            speed: None,
            segments: vec![
                segment(0, "The harbour woke."),
                segment(1, "Mara walked the quay."),
            ],
        }
    }

    async fn new_store(prefix: &str) -> (StudioProjectStore, PathBuf) {
        let root = test_env_root(prefix);
        std::fs::create_dir_all(&root).expect("root should be created");
        let store = initialize_test_store(root.join("test.sqlite3"), root.join("media")).await;
        (store, root)
    }

    async fn insert_speech_record(store: &StudioProjectStore, id: &str) {
        let db = store.db.connection().await.expect("connection");
        db.execute_unprepared(&format!(
            "INSERT INTO speech_history_records (id, created_at, route_kind, input_text, generation_time_ms, audio_duration_secs, audio_mime_type, audio_storage_path) VALUES ('{id}', 0, 'studio', 'text', 1.0, 2.5, 'audio/wav', 'studio/{id}.wav')"
        ))
        .await
        .expect("speech record inserts");
    }

    async fn project(store: &StudioProjectStore, id: &str) -> StudioProjectRecord {
        store
            .get_project(id.to_string())
            .await
            .expect("read should succeed")
            .expect("project should exist")
    }

    fn state_of(project: &StudioProjectRecord, segment_id: &str) -> StudioSegmentRenderState {
        project
            .segments
            .iter()
            .find(|segment| segment.id == segment_id)
            .expect("segment should exist")
            .render_state
    }

    /// Finish the claimed render the way the worker does.
    async fn complete(
        store: &StudioProjectStore,
        claim: &StudioRenderClaim,
        record_id: &str,
    ) -> Vec<String> {
        let current = project(store, &claim.project_id).await;
        let pronunciations = store
            .list_project_pronunciations(claim.project_id.clone())
            .await
            .expect("pronunciations should list");
        let fingerprint = current
            .segments
            .iter()
            .find(|segment| segment.id == claim.segment_id)
            .map(|segment| segment_render_input(&current, segment, &pronunciations).2)
            .unwrap_or_default();
        insert_speech_record(store, record_id).await;
        store
            .complete_render(claim, record_id.to_string(), fingerprint)
            .await
            .expect("render should complete")
    }

    async fn render_now(
        store: &StudioProjectStore,
        project_id: &str,
        segment_id: &str,
        record_id: &str,
    ) {
        store
            .enqueue_render(project_id.to_string(), Some(vec![segment_id.to_string()]))
            .await
            .expect("enqueue should succeed")
            .expect("project should exist");
        let claim = store
            .claim_next_render()
            .await
            .expect("claim should succeed")
            .expect("a segment should be queued");
        assert_eq!(claim.segment_id, segment_id);
        complete(store, &claim, record_id).await;
    }

    #[tokio::test]
    async fn audio_goes_stale_on_edits_and_survives_reorder() {
        let (store, root) = new_store("studio-render-state").await;
        let created = store
            .create_project(narration("Harbour"))
            .await
            .expect("project should be created");
        let (first, second) = (
            created.segments[0].id.clone(),
            created.segments[1].id.clone(),
        );
        render_now(&store, &created.id, &first, "take-1").await;

        let current = project(&store, &created.id).await;
        assert_eq!(
            state_of(&current, &first),
            StudioSegmentRenderState::Rendered
        );
        assert_eq!(
            state_of(&current, &second),
            StudioSegmentRenderState::NotRendered
        );

        let reordered = store
            .reorder_segments(created.id.clone(), vec![second.clone(), first.clone()])
            .await
            .expect("reorder should succeed")
            .expect("project should exist");
        assert_eq!(
            state_of(&reordered, &first),
            StudioSegmentRenderState::Rendered
        );
        assert_eq!(
            reordered.segments[1].speech_record_id.as_deref(),
            Some("take-1")
        );

        let edit = |text: &str| {
            store.update_segment(
                created.id.clone(),
                first.clone(),
                Some(text.to_string()),
                None,
            )
        };
        let edited = edit("The harbour woke early.").await.unwrap().unwrap();
        assert_eq!(state_of(&edited, &first), StudioSegmentRenderState::Stale);
        assert_eq!(
            edited.segments[1].speech_record_id.as_deref(),
            Some("take-1")
        );
        let restored = edit("The harbour woke.").await.unwrap().unwrap();
        assert_eq!(
            state_of(&restored, &first),
            StudioSegmentRenderState::Rendered
        );

        let set_speaker = |speaker: &str| {
            store.update_project(
                created.id.clone(),
                UpdateStudioProjectRecord {
                    speaker: Some(Some(speaker.to_string())),
                    ..UpdateStudioProjectRecord::default()
                },
            )
        };
        let revoiced = set_speaker("am_adam").await.unwrap().unwrap();
        assert_eq!(state_of(&revoiced, &first), StudioSegmentRenderState::Stale);
        let restored = set_speaker("af_heart").await.unwrap().unwrap();
        assert_eq!(
            state_of(&restored, &first),
            StudioSegmentRenderState::Rendered
        );

        let rule = store
            .create_project_pronunciation(
                created.id.clone(),
                NewStudioProjectPronunciationRecord {
                    source_text: "harbour".to_string(),
                    replacement_text: "har-bur".to_string(),
                    locale: None,
                },
            )
            .await
            .unwrap()
            .unwrap();
        let current = project(&store, &created.id).await;
        assert_eq!(state_of(&current, &first), StudioSegmentRenderState::Stale);
        store
            .delete_project_pronunciation(created.id.clone(), rule.id)
            .await
            .unwrap();
        let current = project(&store, &created.id).await;
        assert_eq!(
            state_of(&current, &first),
            StudioSegmentRenderState::Rendered
        );

        let override_voice = store
            .update_segment(
                created.id.clone(),
                first.clone(),
                None,
                Some(Some(StudioSegmentVoiceOverride {
                    model_id: None,
                    voice_mode: StudioProjectVoiceMode::BuiltIn,
                    speaker: Some("bm_george".to_string()),
                    saved_voice_id: None,
                })),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            state_of(&override_voice, &first),
            StudioSegmentRenderState::Stale
        );
        let project_voice = store
            .update_segment(created.id.clone(), first.clone(), None, Some(None))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            state_of(&project_voice, &first),
            StudioSegmentRenderState::Rendered
        );
        assert_eq!(project_voice.segments[1].speaker, None);

        let (summaries, _) = store.list_projects_page(10, None).await.unwrap();
        let summary = &summaries[0];
        assert_eq!(summary.segment_count, 2);
        assert_eq!(summary.rendered_segment_count, 1);
        assert_eq!(summary.stale_segment_count, 0);
        assert_eq!(summary.audio_duration_secs, 2.5);
        let created_empty = store.create_project(narration("Empty")).await.unwrap();
        let (summaries, _) = store.list_projects_page(10, None).await.unwrap();
        let empty = summaries
            .iter()
            .find(|summary| summary.id == created_empty.id)
            .unwrap();
        assert!(empty.audio_duration_secs.is_sign_positive());

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn render_queue_runs_in_order_and_cancel_leaves_the_running_paragraph() {
        let (store, root) = new_store("studio-render-queue").await;
        let created = store.create_project(narration("Queue")).await.unwrap();
        let (first, second) = (
            created.segments[0].id.clone(),
            created.segments[1].id.clone(),
        );

        let job = store
            .enqueue_render(created.id.clone(), None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.queued_segment_ids, vec![first.clone(), second.clone()]);
        let queued = project(&store, &created.id).await;
        assert_eq!(state_of(&queued, &second), StudioSegmentRenderState::Queued);
        assert_eq!(
            queued.active_render_job.as_ref().map(|job| job.id.as_str()),
            Some(job.id.as_str())
        );
        assert!(
            store
                .enqueue_render(created.id.clone(), None)
                .await
                .is_err(),
            "nothing is left to queue"
        );

        let claim = store.claim_next_render().await.unwrap().unwrap();
        assert_eq!(claim.segment_id, first);
        let cancelled = store
            .cancel_render_job(created.id.clone(), job.id.clone())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cancelled.status, StudioProjectRenderJobStatus::Cancelled);
        let current = project(&store, &created.id).await;
        assert_eq!(
            state_of(&current, &first),
            StudioSegmentRenderState::Rendering
        );
        assert_eq!(
            state_of(&current, &second),
            StudioSegmentRenderState::NotRendered
        );
        assert!(current.active_render_job.is_none());

        // A restart mid-render: the paragraph's job is gone, so it leaves
        // the queue instead of waiting forever.
        store.requeue_interrupted_renders().await.unwrap();
        let current = project(&store, &created.id).await;
        assert_eq!(
            state_of(&current, &first),
            StudioSegmentRenderState::NotRendered
        );
        assert!(store.claim_next_render().await.unwrap().is_none());

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn a_restart_requeues_the_paragraph_of_an_active_job() {
        let (store, root) = new_store("studio-render-restart").await;
        let created = store.create_project(narration("Restart")).await.unwrap();
        store
            .enqueue_render(created.id.clone(), None)
            .await
            .unwrap();
        let claim = store.claim_next_render().await.unwrap().unwrap();

        assert_eq!(store.requeue_interrupted_renders().await.unwrap(), 1);
        let again = store.claim_next_render().await.unwrap().unwrap();
        assert_eq!(again, claim);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn failures_finish_the_job_as_failed_and_an_edit_clears_them() {
        let (store, root) = new_store("studio-render-failure").await;
        let created = store.create_project(narration("Failure")).await.unwrap();
        let (first, second) = (
            created.segments[0].id.clone(),
            created.segments[1].id.clone(),
        );
        let job = store
            .enqueue_render(created.id.clone(), None)
            .await
            .unwrap()
            .unwrap();

        let claim = store.claim_next_render().await.unwrap().unwrap();
        store.fail_render(&claim, "voice not found").await.unwrap();
        let claim = store.claim_next_render().await.unwrap().unwrap();
        assert_eq!(claim.segment_id, second);
        complete(&store, &claim, "take-2").await;
        assert!(store.claim_next_render().await.unwrap().is_none());

        let jobs = store
            .list_project_render_jobs(created.id.clone())
            .await
            .unwrap();
        let finished = jobs
            .iter()
            .find(|candidate| candidate.id == job.id)
            .unwrap();
        assert_eq!(finished.status, StudioProjectRenderJobStatus::Failed);
        assert_eq!(
            finished.error_message.as_deref(),
            Some("1 paragraph failed: voice not found")
        );
        let current = project(&store, &created.id).await;
        let failed = current
            .segments
            .iter()
            .find(|segment| segment.id == first)
            .unwrap();
        assert_eq!(failed.render_state, StudioSegmentRenderState::Failed);
        assert_eq!(failed.render_error.as_deref(), Some("voice not found"));

        let edited = store
            .update_segment(
                created.id.clone(),
                first.clone(),
                Some("New words.".to_string()),
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            state_of(&edited, &first),
            StudioSegmentRenderState::NotRendered
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn a_missing_model_fails_the_rest_of_the_job_at_once() {
        let (store, root) = new_store("studio-render-job-failure").await;
        let created = store
            .create_project(narration("Missing model"))
            .await
            .unwrap();
        let job = store
            .enqueue_render(created.id.clone(), None)
            .await
            .unwrap()
            .unwrap();
        let claim = store.claim_next_render().await.unwrap().unwrap();
        store.fail_render(&claim, "not downloaded").await.unwrap();
        store
            .fail_queued_in_job(&claim, "not downloaded")
            .await
            .unwrap();

        assert!(store.claim_next_render().await.unwrap().is_none());
        let jobs = store
            .list_project_render_jobs(created.id.clone())
            .await
            .unwrap();
        let finished = jobs
            .iter()
            .find(|candidate| candidate.id == job.id)
            .unwrap();
        assert_eq!(
            finished.error_message.as_deref(),
            Some("2 paragraphs failed: not downloaded")
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn replaced_and_orphaned_takes_are_released() {
        let (store, root) = new_store("studio-render-release").await;
        let created = store.create_project(narration("Takes")).await.unwrap();
        let (first, second) = (
            created.segments[0].id.clone(),
            created.segments[1].id.clone(),
        );
        render_now(&store, &created.id, &first, "take-1").await;

        store
            .enqueue_render(created.id.clone(), Some(vec![first.clone()]))
            .await
            .unwrap();
        let claim = store.claim_next_render().await.unwrap().unwrap();
        assert_eq!(
            complete(&store, &claim, "take-2").await,
            vec!["take-1".to_string()]
        );

        store
            .enqueue_render(created.id.clone(), Some(vec![second.clone()]))
            .await
            .unwrap();
        let claim = store.claim_next_render().await.unwrap().unwrap();
        let deleted = store
            .delete_segment(created.id.clone(), second.clone())
            .await
            .unwrap()
            .unwrap();
        assert!(deleted.released_speech_record_ids.is_empty());
        assert_eq!(
            complete(&store, &claim, "take-3").await,
            vec!["take-3".to_string()]
        );

        let released = store.delete_project(created.id.clone()).await.unwrap();
        assert_eq!(released, Some(vec!["take-2".to_string()]));

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn audio_from_before_fingerprints_is_stamped_once() {
        let (store, root) = new_store("studio-render-stamp").await;
        let created = store.create_project(narration("Legacy")).await.unwrap();
        let first = created.segments[0].id.clone();
        insert_speech_record(&store, "legacy-take").await;
        let db = store.db.connection().await.unwrap();
        db.execute_unprepared(&format!(
            "UPDATE studio_project_segments SET speech_record_id = 'legacy-take' WHERE id = '{first}'"
        ))
        .await
        .unwrap();
        assert_eq!(
            state_of(&project(&store, &created.id).await, &first),
            StudioSegmentRenderState::Rendered
        );

        assert_eq!(store.stamp_unfingerprinted_renders().await.unwrap(), 1);
        assert_eq!(store.stamp_unfingerprinted_renders().await.unwrap(), 0);
        let revoiced = store
            .update_project(
                created.id.clone(),
                UpdateStudioProjectRecord {
                    speaker: Some(Some("am_adam".to_string())),
                    ..UpdateStudioProjectRecord::default()
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(state_of(&revoiced, &first), StudioSegmentRenderState::Stale);

        let _ = std::fs::remove_dir_all(root);
    }

    fn rule(source: &str, replacement: &str) -> StudioProjectPronunciationRecord {
        StudioProjectPronunciationRecord {
            id: source.to_string(),
            project_id: "project".to_string(),
            source_text: source.to_string(),
            replacement_text: replacement.to_string(),
            locale: None,
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn pronunciations_match_whole_words_in_any_case_once() {
        let rules = [
            rule("SQL", "sequel"),
            rule("sequel", "WRONG"),
            rule("C++", "C plus plus"),
        ];
        assert_eq!(
            apply_pronunciations("sql, SQL and MySQL; also C++.", &rules),
            "sequel, sequel and MySQL; also C plus plus."
        );
        assert_eq!(apply_pronunciations("Izwi", &[]), "Izwi");
        assert_eq!(
            apply_pronunciations(
                "New York and York",
                &[rule("York", "Yawk"), rule("New York", "Noo Yawk")]
            ),
            "Noo Yawk and Yawk"
        );
    }
}

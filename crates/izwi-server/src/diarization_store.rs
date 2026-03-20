//! Persistent diarization history storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout::{self, MediaGroup};

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegmentRecord {
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationWordRecord {
    pub word: String,
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub speaker_confidence: Option<f32>,
    pub overlaps_segment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationUtteranceRecord {
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub word_start: usize,
    pub word_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub speaker_count: usize,
    pub corrected_speaker_count: usize,
    pub speaker_name_override_count: usize,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcript_preview: String,
    pub transcript_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationRecord {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub corrected_speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
    pub speaker_name_overrides: BTreeMap<String, String>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StoredDiarizationAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewDiarizationRecord {
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
    pub speaker_name_overrides: BTreeMap<String, String>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
}

#[derive(Clone)]
pub struct DiarizationStore {
    db_path: PathBuf,
    media_root: PathBuf,
}

impl DiarizationStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare diarization storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!("Failed to open diarization database: {}", db_path.display())
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS diarization_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                model_id TEXT NULL,
                asr_model_id TEXT NULL,
                aligner_model_id TEXT NULL,
                llm_model_id TEXT NULL,
                min_speakers INTEGER NULL,
                max_speakers INTEGER NULL,
                min_speech_duration_ms REAL NULL,
                min_silence_duration_ms REAL NULL,
                enable_llm_refinement INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                duration_secs REAL NULL,
                rtf REAL NULL,
                speaker_count INTEGER NOT NULL,
                alignment_coverage REAL NULL,
                unattributed_words INTEGER NOT NULL,
                llm_refined INTEGER NOT NULL,
                asr_text TEXT NOT NULL,
                raw_transcript TEXT NOT NULL,
                transcript TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                words_json TEXT NOT NULL,
                utterances_json TEXT NOT NULL,
                speaker_name_overrides_json TEXT NOT NULL DEFAULT '{}',
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_diarization_records_created_at
                ON diarization_records(created_at DESC);
            "#,
        )
        .context("Failed to initialize diarization database schema")?;
        ensure_diarization_records_speaker_name_overrides_column(&conn)?;

        Ok(Self {
            db_path,
            media_root,
        })
    }

    pub async fn list_records(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<DiarizationRecordSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    model_id,
                    speaker_count,
                    utterances_json,
                    speaker_name_overrides_json,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    transcript
                FROM diarization_records
                ORDER BY created_at DESC, id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], |row| {
                let speaker_count = row
                    .get::<_, Option<i64>>(3)?
                    .and_then(i64_to_usize)
                    .unwrap_or(0);
                let utterances: Vec<DiarizationUtteranceRecord> = parse_json_vec(row.get(4)?);
                let speaker_name_overrides = parse_json_map(row.get(5)?);
                let transcript: String = row.get(11)?;
                Ok(DiarizationRecordSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    model_id: row.get(2)?,
                    speaker_count,
                    corrected_speaker_count: corrected_speaker_count_from_parts(
                        speaker_count,
                        &[],
                        &[],
                        &utterances,
                        &speaker_name_overrides,
                    ),
                    speaker_name_override_count: speaker_name_overrides.len(),
                    duration_secs: row.get(6)?,
                    processing_time_ms: row.get(7)?,
                    rtf: row.get(8)?,
                    audio_mime_type: row.get(9)?,
                    audio_filename: row.get(10)?,
                    transcript_preview: transcript_preview_with_utterances(
                        &utterances,
                        &speaker_name_overrides,
                        transcript.as_str(),
                    ),
                    transcript_chars: transcript.chars().count(),
                })
            })?;

            let mut records = Vec::new();
            for row in rows {
                records.push(row?);
            }
            Ok(records)
        })
        .await
    }

    pub async fn get_record(&self, record_id: String) -> anyhow::Result<Option<DiarizationRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let record = fetch_record_without_audio(&conn, &record_id)?;
            Ok(record)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<StoredDiarizationAudio>> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_storage_path, audio_mime_type, audio_filename
                    FROM diarization_records
                    WHERE id = ?1
                    "#,
                    params![record_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    },
                )
                .optional()?;
            let Some((audio_storage_path, audio_mime_type, audio_filename)) = audio else {
                return Ok(None);
            };

            let audio_bytes =
                storage_layout::read_media_file(&media_root, audio_storage_path.as_str())?;

            Ok(Some(StoredDiarizationAudio {
                audio_bytes,
                audio_mime_type,
                audio_filename,
            }))
        })
        .await
    }

    pub async fn create_record(
        &self,
        record: NewDiarizationRecord,
    ) -> anyhow::Result<DiarizationRecord> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let record_id = format!("dir_{}", uuid::Uuid::new_v4().simple());

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let asr_model_id = sanitize_optional_text(record.asr_model_id.as_deref(), 160);
            let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
            let llm_model_id = sanitize_optional_text(record.llm_model_id.as_deref(), 160);
            let min_speakers = record
                .min_speakers
                .and_then(|value| i64::try_from(value).ok());
            let max_speakers = record
                .max_speakers
                .and_then(|value| i64::try_from(value).ok());
            let min_speech_duration_ms = record
                .min_speech_duration_ms
                .filter(|value| value.is_finite() && *value >= 0.0);
            let min_silence_duration_ms = record
                .min_silence_duration_ms
                .filter(|value| value.is_finite() && *value >= 0.0);
            let processing_time_ms = if record.processing_time_ms.is_finite() {
                record.processing_time_ms.max(0.0)
            } else {
                0.0
            };
            let duration_secs = record
                .duration_secs
                .filter(|value| value.is_finite() && *value >= 0.0);
            let rtf = record
                .rtf
                .filter(|value| value.is_finite() && *value >= 0.0);
            let speaker_count = i64::try_from(record.speaker_count).unwrap_or(0);
            let alignment_coverage = record
                .alignment_coverage
                .filter(|value| value.is_finite() && *value >= 0.0);
            let unattributed_words = i64::try_from(record.unattributed_words).unwrap_or(0);
            let asr_text = sanitize_required_text(record.asr_text.as_str(), 40_000);
            let raw_transcript = sanitize_required_text(record.raw_transcript.as_str(), 100_000);
            let transcript = sanitize_required_text(record.transcript.as_str(), 100_000);
            let speaker_name_overrides = sanitize_speaker_name_overrides(
                &record.speaker_name_overrides,
                &raw_speaker_labels_from_parts(&record.segments, &record.words, &record.utterances),
            )?;
            let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
            let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);

            if record.audio_bytes.is_empty() {
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            let audio_storage_path = storage_layout::persist_audio_file(
                &media_root,
                MediaGroup::Uploads,
                "diarization",
                &record_id,
                audio_filename.as_deref(),
                audio_mime_type.as_str(),
                &record.audio_bytes,
            )?;

            let segments_json =
                serde_json::to_string(&record.segments).context("Failed serializing segments")?;
            let words_json =
                serde_json::to_string(&record.words).context("Failed serializing words")?;
            let utterances_json = serde_json::to_string(&record.utterances)
                .context("Failed serializing utterances")?;
            let speaker_name_overrides_json = serde_json::to_string(&speaker_name_overrides)
                .context("Failed serializing speaker name overrides")?;

            if let Err(err) = conn.execute(
                r#"
                INSERT INTO diarization_records (
                    id,
                    created_at,
                    model_id,
                    asr_model_id,
                    aligner_model_id,
                    llm_model_id,
                    min_speakers,
                    max_speakers,
                    min_speech_duration_ms,
                    min_silence_duration_ms,
                    enable_llm_refinement,
                    processing_time_ms,
                    duration_secs,
                    rtf,
                    speaker_count,
                    alignment_coverage,
                    unattributed_words,
                    llm_refined,
                    asr_text,
                    raw_transcript,
                    transcript,
                    segments_json,
                    words_json,
                    utterances_json,
                    speaker_name_overrides_json,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path
                )
                VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                    ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28
                )
                "#,
                params![
                    &record_id,
                    now,
                    model_id,
                    asr_model_id,
                    aligner_model_id,
                    llm_model_id,
                    min_speakers,
                    max_speakers,
                    min_speech_duration_ms,
                    min_silence_duration_ms,
                    if record.enable_llm_refinement {
                        1_i64
                    } else {
                        0_i64
                    },
                    processing_time_ms,
                    duration_secs,
                    rtf,
                    speaker_count,
                    alignment_coverage,
                    unattributed_words,
                    if record.llm_refined { 1_i64 } else { 0_i64 },
                    asr_text,
                    raw_transcript,
                    transcript,
                    segments_json,
                    words_json,
                    utterances_json,
                    speaker_name_overrides_json,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path
                ],
            ) {
                let _ = storage_layout::delete_media_file(
                    &media_root,
                    Some(audio_storage_path.as_str()),
                );
                return Err(err).context("Failed to insert diarization record");
            }

            let created = fetch_record_without_audio(&conn, &record_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created diarization record"))?;
            Ok(created)
        })
        .await
    }

    pub async fn delete_record(&self, record_id: String) -> anyhow::Result<bool> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio_storage_path = conn
                .query_row(
                    "SELECT audio_storage_path FROM diarization_records WHERE id = ?1",
                    params![&record_id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten();
            let changed = conn.execute(
                "DELETE FROM diarization_records WHERE id = ?1",
                params![record_id],
            )?;

            if changed > 0 {
                storage_layout::delete_media_file(&media_root, audio_storage_path.as_deref())?;
            }

            Ok(changed > 0)
        })
        .await
    }

    pub async fn update_speaker_name_overrides(
        &self,
        record_id: String,
        speaker_name_overrides: BTreeMap<String, String>,
    ) -> anyhow::Result<Option<DiarizationRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let existing = fetch_record_without_audio(&conn, &record_id)?;
            let Some(existing) = existing else {
                return Ok(None);
            };

            let sanitized_overrides = sanitize_speaker_name_overrides(
                &speaker_name_overrides,
                &raw_speaker_labels_from_parts(
                    &existing.segments,
                    &existing.words,
                    &existing.utterances,
                ),
            )?;
            let speaker_name_overrides_json = serde_json::to_string(&sanitized_overrides)
                .context("Failed serializing speaker name overrides")?;

            conn.execute(
                r#"
                UPDATE diarization_records
                SET speaker_name_overrides_json = ?2
                WHERE id = ?1
                "#,
                params![record_id, speaker_name_overrides_json],
            )
            .context("Failed updating speaker name overrides")?;

            fetch_record_without_audio(&conn, &existing.id)
        })
        .await
    }

    async fn run_blocking<F, T>(&self, task_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || task_fn(db_path))
            .await
            .map_err(|err| anyhow!("Diarization storage worker failed: {err}"))?
    }
}

fn fetch_record_without_audio(
    conn: &Connection,
    record_id: &str,
) -> anyhow::Result<Option<DiarizationRecord>> {
    let record = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                model_id,
                asr_model_id,
                aligner_model_id,
                llm_model_id,
                min_speakers,
                max_speakers,
                min_speech_duration_ms,
                min_silence_duration_ms,
                enable_llm_refinement,
                processing_time_ms,
                duration_secs,
                rtf,
                speaker_count,
                alignment_coverage,
                unattributed_words,
                llm_refined,
                asr_text,
                raw_transcript,
                transcript,
                segments_json,
                words_json,
                utterances_json,
                speaker_name_overrides_json,
                audio_mime_type,
                audio_filename
            FROM diarization_records
            WHERE id = ?1
            "#,
            params![record_id],
            map_diarization_record,
        )
        .optional()?;
    Ok(record)
}

fn map_diarization_record(row: &Row<'_>) -> rusqlite::Result<DiarizationRecord> {
    let min_speakers = row.get::<_, Option<i64>>(6)?.and_then(i64_to_usize);
    let max_speakers = row.get::<_, Option<i64>>(7)?.and_then(i64_to_usize);
    let speaker_count = row
        .get::<_, Option<i64>>(14)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let unattributed_words = row
        .get::<_, Option<i64>>(16)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let segments_raw: String = row.get(21)?;
    let words_raw: String = row.get(22)?;
    let utterances_raw: String = row.get(23)?;
    let speaker_name_overrides = parse_json_map(row.get(24)?);
    let segments: Vec<DiarizationSegmentRecord> = parse_json_vec(Some(segments_raw));
    let words: Vec<DiarizationWordRecord> = parse_json_vec(Some(words_raw));
    let utterances: Vec<DiarizationUtteranceRecord> = parse_json_vec(Some(utterances_raw));
    let corrected_speaker_count = corrected_speaker_count_from_parts(
        speaker_count,
        &segments,
        &words,
        &utterances,
        &speaker_name_overrides,
    );

    Ok(DiarizationRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        model_id: row.get(2)?,
        asr_model_id: row.get(3)?,
        aligner_model_id: row.get(4)?,
        llm_model_id: row.get(5)?,
        min_speakers,
        max_speakers,
        min_speech_duration_ms: row.get(8)?,
        min_silence_duration_ms: row.get(9)?,
        enable_llm_refinement: row.get::<_, i64>(10)? > 0,
        processing_time_ms: row.get(11)?,
        duration_secs: row.get(12)?,
        rtf: row.get(13)?,
        speaker_count,
        corrected_speaker_count,
        alignment_coverage: row.get(15)?,
        unattributed_words,
        llm_refined: row.get::<_, i64>(17)? > 0,
        asr_text: row.get(18)?,
        raw_transcript: row.get(19)?,
        transcript: row.get(20)?,
        segments,
        words,
        utterances,
        speaker_name_overrides,
        audio_mime_type: row.get(25)?,
        audio_filename: row.get(26)?,
    })
}

fn parse_json_vec<T>(raw: Option<String>) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    raw.and_then(|value| serde_json::from_str::<Vec<T>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn parse_json_map(raw: Option<String>) -> BTreeMap<String, String> {
    raw.and_then(|value| serde_json::from_str::<BTreeMap<String, String>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn transcript_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No transcript".to_string();
    }
    truncate_string(&normalized, 180)
}

fn transcript_preview_with_utterances(
    utterances: &[DiarizationUtteranceRecord],
    speaker_name_overrides: &BTreeMap<String, String>,
    transcript: &str,
) -> String {
    if utterances.is_empty() {
        return transcript_preview(transcript);
    }

    let normalized = utterances
        .iter()
        .filter_map(|utterance| {
            let text = utterance.text.trim();
            if text.is_empty() {
                return None;
            }
            Some(format!(
                "{} [{:.2}s - {:.2}s]: {}",
                resolve_speaker_label(utterance.speaker.as_str(), speaker_name_overrides),
                utterance.start,
                utterance.end,
                text
            ))
        })
        .collect::<Vec<_>>()
        .join(" ");

    if normalized.is_empty() {
        transcript_preview(transcript)
    } else {
        truncate_string(&normalized, 180)
    }
}

fn sanitize_required_text(raw: &str, max_chars: usize) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        " ".to_string()
    } else {
        truncate_string(trimmed, max_chars)
    }
}

fn sanitize_optional_text(raw: Option<&str>, max_chars: usize) -> Option<String> {
    let normalized = raw
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(truncate_string(&normalized, max_chars))
    }
}

fn sanitize_audio_mime_type(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "audio/wav".to_string()
    } else {
        truncate_string(trimmed, 80)
    }
}

fn ensure_diarization_records_speaker_name_overrides_column(
    conn: &Connection,
) -> anyhow::Result<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(diarization_records)")
        .context("Failed to inspect diarization_records schema")?;
    let mut rows = stmt
        .query([])
        .context("Failed to query diarization_records schema info")?;

    while let Some(row) = rows
        .next()
        .context("Failed reading diarization_records schema row")?
    {
        let name: String = row
            .get(1)
            .context("Failed reading diarization_records column name")?;
        if name == "speaker_name_overrides_json" {
            return Ok(());
        }
    }

    conn.execute(
        "ALTER TABLE diarization_records ADD COLUMN speaker_name_overrides_json TEXT NOT NULL DEFAULT '{}'",
        [],
    )
    .context("Failed adding diarization_records.speaker_name_overrides_json column")?;
    Ok(())
}

fn resolve_speaker_label(
    raw_label: &str,
    speaker_name_overrides: &BTreeMap<String, String>,
) -> String {
    speaker_name_overrides
        .get(raw_label)
        .cloned()
        .unwrap_or_else(|| raw_label.to_string())
}

fn raw_speaker_labels_from_parts(
    segments: &[DiarizationSegmentRecord],
    words: &[DiarizationWordRecord],
    utterances: &[DiarizationUtteranceRecord],
) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();

    for segment in segments {
        let trimmed = segment.speaker.trim();
        if !trimmed.is_empty() {
            labels.insert(trimmed.to_string());
        }
    }

    for word in words {
        let trimmed = word.speaker.trim();
        if !trimmed.is_empty() {
            labels.insert(trimmed.to_string());
        }
    }

    for utterance in utterances {
        let trimmed = utterance.speaker.trim();
        if !trimmed.is_empty() {
            labels.insert(trimmed.to_string());
        }
    }

    labels
}

fn corrected_speaker_count_from_parts(
    raw_speaker_count: usize,
    segments: &[DiarizationSegmentRecord],
    words: &[DiarizationWordRecord],
    utterances: &[DiarizationUtteranceRecord],
    speaker_name_overrides: &BTreeMap<String, String>,
) -> usize {
    let raw_labels = raw_speaker_labels_from_parts(segments, words, utterances);
    if raw_labels.is_empty() {
        return raw_speaker_count;
    }

    raw_labels
        .into_iter()
        .map(|label| resolve_speaker_label(label.as_str(), speaker_name_overrides))
        .collect::<BTreeSet<_>>()
        .len()
}

fn sanitize_speaker_name_overrides(
    overrides: &BTreeMap<String, String>,
    known_labels: &BTreeSet<String>,
) -> anyhow::Result<BTreeMap<String, String>> {
    let mut sanitized = BTreeMap::new();

    for (raw_label, corrected_label) in overrides {
        let raw = sanitize_required_text(raw_label.as_str(), 80);
        if !known_labels.contains(&raw) {
            return Err(anyhow!("Unknown diarization speaker label: {raw}"));
        }

        let normalized = corrected_label
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if normalized.is_empty() {
            continue;
        }

        let corrected = truncate_string(normalized.as_str(), 80);
        if corrected != raw {
            sanitized.insert(raw, corrected);
        }
    }

    Ok(sanitized)
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        result.push(ch);
    }
    if input.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

fn now_unix_millis_i64() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

fn i64_to_u64(value: i64) -> u64 {
    if value.is_negative() {
        0
    } else {
        value as u64
    }
}

fn i64_to_usize(value: i64) -> Option<usize> {
    if value.is_negative() {
        None
    } else {
        usize::try_from(value).ok()
    }
}

#[allow(dead_code)]
pub const fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_store() -> (DiarizationStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "izwi-diarization-store-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let store = DiarizationStore::initialize_at(root.join("izwi.sqlite3"), root.join("media"))
            .expect("test store should initialize");
        (store, root)
    }

    fn sample_record() -> NewDiarizationRecord {
        NewDiarizationRecord {
            model_id: Some("diar_streaming_sortformer_4spk-v2.1".to_string()),
            asr_model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            llm_model_id: Some("Qwen3-1.7B-GGUF".to_string()),
            min_speakers: Some(1),
            max_speakers: Some(4),
            min_speech_duration_ms: Some(240.0),
            min_silence_duration_ms: Some(200.0),
            enable_llm_refinement: true,
            processing_time_ms: 500.0,
            duration_secs: Some(12.0),
            rtf: Some(0.4),
            speaker_count: 2,
            alignment_coverage: Some(0.95),
            unattributed_words: 0,
            llm_refined: true,
            asr_text: "hello there".to_string(),
            raw_transcript: "SPEAKER_00 [0.00s - 1.20s]: Hello there.".to_string(),
            transcript: "SPEAKER_00 [0.00s - 1.20s]: Hello there.".to_string(),
            segments: vec![
                DiarizationSegmentRecord {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.2,
                    confidence: Some(0.9),
                },
                DiarizationSegmentRecord {
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.2,
                    end: 2.5,
                    confidence: Some(0.88),
                },
            ],
            words: vec![
                DiarizationWordRecord {
                    word: "Hello".to_string(),
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 0.4,
                    speaker_confidence: Some(0.92),
                    overlaps_segment: true,
                },
                DiarizationWordRecord {
                    word: "there".to_string(),
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.3,
                    end: 1.8,
                    speaker_confidence: Some(0.89),
                    overlaps_segment: true,
                },
            ],
            utterances: vec![
                DiarizationUtteranceRecord {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.2,
                    text: "Hello there.".to_string(),
                    word_start: 0,
                    word_end: 0,
                },
                DiarizationUtteranceRecord {
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.2,
                    end: 2.5,
                    text: "Hi back.".to_string(),
                    word_start: 1,
                    word_end: 1,
                },
            ],
            speaker_name_overrides: BTreeMap::new(),
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("meeting.wav".to_string()),
            audio_bytes: vec![0_u8, 1_u8, 2_u8, 3_u8],
        }
    }

    #[tokio::test]
    async fn persists_speaker_name_overrides_and_corrected_summary() {
        let (store, root) = build_test_store();

        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");
        assert_eq!(created.corrected_speaker_count, 2);
        assert!(created.speaker_name_overrides.is_empty());

        let updated = store
            .update_speaker_name_overrides(
                created.id.clone(),
                BTreeMap::from([
                    ("SPEAKER_00".to_string(), "Alice".to_string()),
                    ("SPEAKER_01".to_string(), "Alice".to_string()),
                ]),
            )
            .await
            .expect("update should succeed")
            .expect("record should exist");

        assert_eq!(updated.corrected_speaker_count, 1);
        assert_eq!(
            updated.speaker_name_overrides.get("SPEAKER_00"),
            Some(&"Alice".to_string())
        );

        let summaries = store.list_records(10).await.expect("list should succeed");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].speaker_count, 2);
        assert_eq!(summaries[0].corrected_speaker_count, 1);
        assert_eq!(summaries[0].speaker_name_override_count, 2);
        assert!(summaries[0].transcript_preview.contains("Alice"));

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn rejects_unknown_speaker_overrides() {
        let (store, root) = build_test_store();
        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");

        let err = store
            .update_speaker_name_overrides(
                created.id,
                BTreeMap::from([("UNKNOWN".to_string(), "Alice".to_string())]),
            )
            .await
            .expect_err("unknown speaker should fail");

        assert!(err
            .to_string()
            .contains("Unknown diarization speaker label"));
        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }
}

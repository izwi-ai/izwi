//! Persistent saved voice storage backed by SQLite.

use anyhow::{anyhow, Context};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaStorageProvider};
use sea_orm::{ConnectionTrait, EntityTrait, QueryResult, Set, SqlErr};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::StoreError;
use crate::{
    db::{raw, StoreDatabase},
    entity::saved_voices,
    ids::new_uuid,
    persistence::{
        delete_media_object, persist_audio_object, read_media_object, LocalMediaStorageProvider,
    },
    storage_layout,
};

const DEFAULT_PERMISSION_SCOPE: &str = "local_owner";
const DEFAULT_CONSENT_STATUS: &str = "granted";
const LOCAL_TTS_ALLOWED_USE: &str = "local_tts";
const DEFAULT_ALLOWED_USES_JSON: &str = r#"["local_tts","voice_clone_reference"]"#;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SavedVoiceSourceRouteKind {
    VoiceDesign,
    VoiceCloning,
}

impl SavedVoiceSourceRouteKind {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::VoiceDesign => "voice_design",
            Self::VoiceCloning => "voice_cloning",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "voice_design" => Some(Self::VoiceDesign),
            "voice_cloning" => Some(Self::VoiceCloning),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SavedVoiceSummary {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub reference_text_preview: String,
    pub reference_text_chars: usize,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    pub source_record_id: Option<String>,
    pub permission_scope: String,
    pub consent_status: String,
    pub allowed_uses: Vec<String>,
    pub permission_provenance: Option<String>,
    pub permission_revoked_at: Option<u64>,
    /// The language the reference is spoken in, for models that clone
    /// across languages (Maya 2 Native); absent, they read it from the
    /// transcript.
    pub reference_language: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SavedVoice {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    /// What the reference says; a voice for a model that clones from audio
    /// alone (Chatterbox) may have none.
    pub reference_text: Option<String>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    pub source_record_id: Option<String>,
    pub permission_scope: String,
    pub consent_status: String,
    pub allowed_uses: Vec<String>,
    pub permission_provenance: Option<String>,
    pub permission_revoked_at: Option<u64>,
    /// The language the reference is spoken in, for models that clone
    /// across languages (Maya 2 Native); absent, they read it from the
    /// transcript.
    pub reference_language: Option<String>,
}

impl SavedVoice {
    pub fn allows_local_tts_use(&self) -> bool {
        self.permission_revoked_at.is_none()
            && self.consent_status == DEFAULT_CONSENT_STATUS
            && self
                .allowed_uses
                .iter()
                .any(|use_case| use_case == LOCAL_TTS_ALLOWED_USE)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SavedVoiceListCursor {
    pub updated_at: u64,
    pub created_at: u64,
    pub id: String,
}

#[derive(Debug, Clone)]
pub struct StoredSavedVoiceAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_storage_path: String,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewSavedVoice {
    pub name: String,
    pub reference_text: Option<String>,
    pub reference_language: Option<String>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    pub source_record_id: Option<String>,
}

#[derive(Clone)]
pub struct SavedVoiceStore {
    db: StoreDatabase,
    media_storage: Arc<dyn MediaStorageProvider>,
}

impl SavedVoiceStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare saved voice storage layout")?;

        Ok(Self {
            db: StoreDatabase::new(db_path),
            media_storage: Arc::new(LocalMediaStorageProvider::new(media_root)),
        })
    }

    pub fn initialize_with_storage(
        db: StoreDatabase,
        media_storage: Arc<dyn MediaStorageProvider>,
    ) -> Self {
        Self { db, media_storage }
    }

    /// A page of saved voices, newest first; `search` matches the name or
    /// the reference text, ignoring case.
    pub async fn list_voices_page(
        &self,
        limit: usize,
        cursor: Option<SavedVoiceListCursor>,
        search: Option<&str>,
    ) -> anyhow::Result<(Vec<SavedVoiceSummary>, Option<SavedVoiceListCursor>)> {
        let db = self.db.connection().await?;
        let list_limit = i64::try_from(limit.clamp(1, 2000).max(1)).unwrap_or(500);
        let page_size = usize::try_from(list_limit).unwrap_or(500);
        let fetch_limit = list_limit.saturating_add(1);

        let (cursor_updated_at, cursor_created_at, cursor_id) = match cursor {
            Some(cursor) => (
                Some(i64::try_from(cursor.updated_at).unwrap_or(i64::MAX)),
                Some(i64::try_from(cursor.created_at).unwrap_or(i64::MAX)),
                Some(cursor.id),
            ),
            None => (None, None, None),
        };
        let search = search
            .map(str::trim)
            .filter(|search| !search.is_empty())
            .map(raw::like_contains_pattern);
        let rows = db
            .query_all_raw(raw::statement(
                db,
                SAVED_VOICE_PAGE_SQL,
                vec![
                    search.into(),
                    cursor_updated_at.into(),
                    cursor_created_at.into(),
                    cursor_id.into(),
                    fetch_limit.into(),
                ],
            )?)
            .await
        .context("Failed to list saved voices")?;

        let mut records = rows
            .iter()
            .map(map_saved_voice_summary)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let has_more = records.len() > page_size;
        if has_more {
            records.truncate(page_size);
        }

        let next_cursor = if has_more {
            records.last().map(|record| SavedVoiceListCursor {
                updated_at: record.updated_at,
                created_at: record.created_at,
                id: record.id.clone(),
            })
        } else {
            None
        };

        Ok((records, next_cursor))
    }

    pub async fn get_voice(&self, voice_id: String) -> anyhow::Result<Option<SavedVoice>> {
        let db = self.db.connection().await?;
        fetch_voice_without_audio(db, &voice_id).await
    }

    pub async fn get_audio(
        &self,
        voice_id: String,
    ) -> anyhow::Result<Option<StoredSavedVoiceAudio>> {
        let db = self.db.connection().await?;
        let audio = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT audio_storage_path, audio_mime_type, audio_filename
                FROM saved_voices
                WHERE id = ?1
                "#,
                vec![voice_id.into()],
            )?)
            .await
            .context("Failed to load saved voice audio metadata")?;

        let Some(row) = audio else {
            return Ok(None);
        };

        let audio_storage_path: String = row.try_get_by_index(0)?;
        let audio_mime_type: String = row.try_get_by_index(1)?;
        let audio_filename: Option<String> = row.try_get_by_index(2)?;
        // A voice whose recording is gone (deleted or moved by hand) has no
        // audio: callers answer "not found", not a server error.
        let audio = match read_media_object(&self.media_storage, audio_storage_path.as_str()).await
        {
            Ok(audio) => audio,
            Err(err) if err.is_not_found() => {
                tracing::warn!(path = %audio_storage_path, "saved voice audio is missing");
                return Ok(None);
            }
            Err(err) => {
                return Err(anyhow::Error::new(err).context("Failed to read saved voice media"))
            }
        };

        Ok(Some(StoredSavedVoiceAudio {
            audio_bytes: audio.bytes,
            audio_storage_path,
            audio_mime_type,
            audio_filename,
        }))
    }

    pub async fn create_voice(&self, voice: NewSavedVoice) -> anyhow::Result<SavedVoice> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let voice_id = new_uuid();
        let name = sanitize_required_text(voice.name.as_str(), 120, "name")?;
        let reference_text = voice
            .reference_text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .map(|text| sanitize_required_text(text, 4_000, "reference_text"))
            .transpose()?;
        let reference_language = sanitize_optional_text(voice.reference_language.as_deref(), 16);
        let audio_mime_type = sanitize_audio_mime_type(voice.audio_mime_type.as_str());
        let audio_filename = sanitize_optional_text(voice.audio_filename.as_deref(), 260);
        let source_route_kind = voice
            .source_route_kind
            .map(|kind| kind.as_db_value().to_string());
        let source_record_id = sanitize_optional_text(voice.source_record_id.as_deref(), 200);
        let permission_provenance = source_route_kind
            .as_ref()
            .zip(source_record_id.as_ref())
            .map(|(kind, record_id)| format!("{kind}:{record_id}"))
            .or_else(|| Some("local_upload".to_string()));

        if voice.audio_bytes.is_empty() {
            return Err(StoreError::invalid("Audio payload cannot be empty").into());
        }

        let audio_storage_path = persist_audio_object(
            &self.media_storage,
            MediaNamespace::SavedVoice,
            &voice_id,
            audio_filename.as_deref(),
            audio_mime_type.as_str(),
            &voice.audio_bytes,
            HookMetadata::new(),
        )
        .await?;

        if let Err(err) = saved_voices::Entity::insert(saved_voices::ActiveModel {
            id: Set(voice_id.clone()),
            created_at: Set(now),
            updated_at: Set(now),
            name: Set(name),
            reference_text: Set(reference_text),
            audio_mime_type: Set(audio_mime_type),
            audio_filename: Set(audio_filename),
            audio_storage_path: Set(audio_storage_path.clone()),
            source_route_kind: Set(source_route_kind),
            source_record_id: Set(source_record_id),
            permission_scope: Set(DEFAULT_PERMISSION_SCOPE.to_string()),
            consent_status: Set(DEFAULT_CONSENT_STATUS.to_string()),
            allowed_uses_json: Set(DEFAULT_ALLOWED_USES_JSON.to_string()),
            permission_provenance: Set(permission_provenance),
            permission_revoked_at: Set(None),
            reference_language: Set(reference_language),
        })
        .exec(db)
        .await
        {
            let _ = delete_media_object(&self.media_storage, Some(&audio_storage_path)).await;
            // The id is new, so the name is the only unique value that can clash.
            if matches!(err.sql_err(), Some(SqlErr::UniqueConstraintViolation(_))) {
                return Err(StoreError::invalid(
                    "A saved voice with this name already exists. Choose a different name.",
                )
                .into());
            }
            return Err(err).context("Failed to insert saved voice");
        }

        fetch_voice_without_audio(db, &voice_id)
            .await?
            .ok_or_else(|| anyhow!("Failed to fetch created saved voice"))
    }

    pub async fn delete_voice(&self, voice_id: String) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let audio_storage_path = db
            .query_one_raw(raw::statement(
                db,
                "SELECT audio_storage_path FROM saved_voices WHERE id = ?1",
                vec![voice_id.clone().into()],
            )?)
            .await
            .context("Failed to load saved voice media path")?
            .map(|row| row.try_get_by_index::<Option<String>>(0))
            .transpose()?
            .flatten();

        let result = saved_voices::Entity::delete_by_id(voice_id)
            .exec(db)
            .await
            .context("Failed to delete saved voice")?;

        if result.rows_affected > 0 {
            delete_media_object(&self.media_storage, audio_storage_path.as_deref()).await?;
        }

        Ok(result.rows_affected > 0)
    }
}

const SAVED_VOICE_COLUMNS: &str = r#"
    id,
    created_at,
    updated_at,
    name,
    reference_text,
    audio_mime_type,
    audio_filename,
    source_route_kind,
    source_record_id,
    permission_scope,
    consent_status,
    allowed_uses_json,
    permission_provenance,
    permission_revoked_at,
    reference_language
"#;

/// ?1 search pattern (or NULL), ?2-?4 cursor (or NULL), ?5 limit.
const SAVED_VOICE_PAGE_SQL: &str = r#"
    SELECT
        id,
        created_at,
        updated_at,
        name,
        reference_text,
        audio_mime_type,
        audio_filename,
        source_route_kind,
        source_record_id,
        permission_scope,
        consent_status,
        allowed_uses_json,
        permission_provenance,
        permission_revoked_at,
        reference_language
    FROM saved_voices
    WHERE (
        ?1 IS NULL
        OR LOWER(name) LIKE LOWER(?1) ESCAPE '\'
        OR LOWER(reference_text) LIKE LOWER(?1) ESCAPE '\'
    )
    AND (
        ?2 IS NULL
        OR updated_at < ?2
        OR (
            updated_at = ?2
            AND (
                created_at < ?3
                OR (created_at = ?3 AND id < ?4)
            )
        )
    )
    ORDER BY updated_at DESC, created_at DESC, id DESC
    LIMIT ?5
"#;

async fn fetch_voice_without_audio(
    db: &sea_orm::DatabaseConnection,
    voice_id: &str,
) -> anyhow::Result<Option<SavedVoice>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            format!("SELECT {SAVED_VOICE_COLUMNS} FROM saved_voices WHERE id = ?1"),
            vec![voice_id.into()],
        )?)
        .await
        .context("Failed to load saved voice")?;
    row.as_ref().map(map_saved_voice).transpose()
}

fn map_saved_voice(row: &QueryResult) -> anyhow::Result<SavedVoice> {
    let source_route_raw: Option<String> = row.try_get_by_index(7)?;
    let allowed_uses_raw: String = row.try_get_by_index(11)?;
    Ok(SavedVoice {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        updated_at: i64_to_u64(row.try_get_by_index(2)?),
        name: row.try_get_by_index(3)?,
        reference_text: row.try_get_by_index(4)?,
        audio_mime_type: row.try_get_by_index(5)?,
        audio_filename: row.try_get_by_index(6)?,
        source_route_kind: source_route_raw
            .as_deref()
            .and_then(SavedVoiceSourceRouteKind::from_db_value),
        source_record_id: row.try_get_by_index(8)?,
        permission_scope: row.try_get_by_index(9)?,
        consent_status: row.try_get_by_index(10)?,
        allowed_uses: parse_allowed_uses(allowed_uses_raw.as_str()),
        permission_provenance: row.try_get_by_index(12)?,
        permission_revoked_at: row.try_get_by_index::<Option<i64>>(13)?.map(i64_to_u64),
        reference_language: row.try_get_by_index(14)?,
    })
}

fn map_saved_voice_summary(row: &QueryResult) -> anyhow::Result<SavedVoiceSummary> {
    let reference_text: String = row
        .try_get_by_index::<Option<String>>(4)?
        .unwrap_or_default();
    let source_route_raw: Option<String> = row.try_get_by_index(7)?;
    let allowed_uses_raw: String = row.try_get_by_index(11)?;

    Ok(SavedVoiceSummary {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        updated_at: i64_to_u64(row.try_get_by_index(2)?),
        name: row.try_get_by_index(3)?,
        reference_text_preview: reference_text_preview(reference_text.as_str()),
        reference_text_chars: reference_text.chars().count(),
        audio_mime_type: row.try_get_by_index(5)?,
        audio_filename: row.try_get_by_index(6)?,
        source_route_kind: source_route_raw
            .as_deref()
            .and_then(SavedVoiceSourceRouteKind::from_db_value),
        source_record_id: row.try_get_by_index(8)?,
        permission_scope: row.try_get_by_index(9)?,
        consent_status: row.try_get_by_index(10)?,
        allowed_uses: parse_allowed_uses(allowed_uses_raw.as_str()),
        permission_provenance: row.try_get_by_index(12)?,
        permission_revoked_at: row.try_get_by_index::<Option<i64>>(13)?.map(i64_to_u64),
        reference_language: row.try_get_by_index(14)?,
    })
}

fn parse_allowed_uses(raw: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw).unwrap_or_default()
}

fn sanitize_required_text(raw: &str, max_chars: usize, field_name: &str) -> anyhow::Result<String> {
    let normalized = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return Err(StoreError::invalid(format!("{field_name} cannot be empty")).into());
    }
    Ok(truncate_string(&normalized, max_chars))
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

fn reference_text_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No reference text".to_string();
    }
    truncate_string(&normalized, 140)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use sea_orm::ConnectionTrait;

    fn setup_store() -> (tempfile::TempDir, SavedVoiceStore) {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("saved-voices.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);
        let store = SavedVoiceStore::initialize().expect("store");
        (temp_dir, store)
    }

    fn clear_env() {
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    /// DUR-F13: a duplicate name is a typed client error, found by the
    /// database's constraint kind rather than by its message text. Runs on
    /// PostgreSQL too when `IZWI_TEST_STORE_PG_URL` is set.
    #[tokio::test]
    async fn duplicate_name_is_a_typed_invalid_request() {
        let database = crate::db::test_database::TestDatabase::new().await;
        let media = tempfile::tempdir().expect("temp dir");
        let store = SavedVoiceStore::initialize_with_storage(
            database.database(),
            Arc::new(LocalMediaStorageProvider::new(media.path().to_path_buf())),
        );
        let voice = |name: &str| NewSavedVoice {
            name: name.to_string(),
            reference_text: Some("A calm reference sample".to_string()),
            reference_language: None,
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: None,
            audio_bytes: vec![1, 2, 3, 4],
            source_route_kind: None,
            source_record_id: None,
        };
        store
            .create_voice(voice("Demo"))
            .await
            .expect("first voice");
        let error = store
            .create_voice(voice("demo"))
            .await
            .expect_err("names are unique regardless of case");
        assert!(
            crate::error::StoreError::find(&error).is_some(),
            "{error:#}"
        );
        let api = crate::error::store_api_error(error, "Saved voice storage error");
        assert_eq!(api.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(api.message.contains("already exists"), "{}", api.message);
    }

    #[tokio::test]
    async fn a_voice_may_omit_its_transcript() {
        let _guard = env_lock();
        let (_temp, store) = setup_store();
        let voice = |name: &str, transcript: Option<&str>| NewSavedVoice {
            name: name.to_string(),
            reference_text: transcript.map(str::to_string),
            reference_language: None,
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: None,
            audio_bytes: vec![1, 2, 3, 4],
            source_route_kind: None,
            source_record_id: None,
        };
        let without = store
            .create_voice(voice("Audio only", None))
            .await
            .expect("a voice without a transcript");
        assert_eq!(without.reference_text, None);
        let blank = store
            .create_voice(voice("Blank", Some("   ")))
            .await
            .expect("a blank transcript is none");
        assert_eq!(blank.reference_text, None);
        let read = store
            .get_voice(without.id.clone())
            .await
            .expect("voice reads")
            .expect("voice exists");
        assert_eq!(read.reference_text, None);
        let (voices, _) = store
            .list_voices_page(10, None, None)
            .await
            .expect("voices list");
        let listed = voices
            .iter()
            .find(|voice| voice.id == without.id)
            .expect("listed");
        assert_eq!(listed.reference_text_preview, "No reference text");
        assert_eq!(listed.reference_text_chars, 0);
    }

    #[tokio::test]
    async fn creates_lists_reads_audio_and_deletes_voice() {
        let _guard = env_lock();
        let (_temp, store) = setup_store();

        let created = store
            .create_voice(NewSavedVoice {
                name: "  Demo Voice ".to_string(),
                reference_text: Some(" A calm reference sample ".to_string()),
                reference_language: Some("hi".to_string()),
                audio_mime_type: "audio/wav".to_string(),
                audio_filename: Some("demo.wav".to_string()),
                audio_bytes: vec![1, 2, 3, 4],
                source_route_kind: Some(SavedVoiceSourceRouteKind::VoiceCloning),
                source_record_id: Some("speech-1".to_string()),
            })
            .await
            .expect("voice should create");

        assert_eq!(created.name, "Demo Voice");
        assert_eq!(created.reference_language.as_deref(), Some("hi"));
        assert_eq!(
            created.source_route_kind,
            Some(SavedVoiceSourceRouteKind::VoiceCloning)
        );
        assert_eq!(created.permission_scope, "local_owner");
        assert_eq!(created.consent_status, "granted");
        assert_eq!(
            created.allowed_uses,
            vec!["local_tts".to_string(), "voice_clone_reference".to_string()]
        );
        assert_eq!(
            created.permission_provenance.as_deref(),
            Some("voice_cloning:speech-1")
        );
        assert!(created.permission_revoked_at.is_none());
        assert!(created.allows_local_tts_use());

        let audio = store
            .get_audio(created.id.clone())
            .await
            .expect("audio should load")
            .expect("audio should exist");
        assert_eq!(audio.audio_bytes, vec![1, 2, 3, 4]);

        let (voices, cursor) = store
            .list_voices_page(10, None, None)
            .await
            .expect("voices should list");
        assert!(cursor.is_none());
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0].name, "Demo Voice");
        assert_eq!(voices[0].reference_language.as_deref(), Some("hi"));
        assert_eq!(voices[0].permission_scope, "local_owner");
        assert!(voices[0]
            .allowed_uses
            .iter()
            .any(|allowed_use| allowed_use == "local_tts"));

        let db = store.db.connection().await.expect("db connection");
        let revoke_statement = raw::statement(
            db,
            "UPDATE saved_voices SET permission_revoked_at = 1 WHERE id = ?1",
            vec![created.id.clone().into()],
        )
        .expect("revoke statement");
        db.execute_raw(revoke_statement)
            .await
            .expect("voice permission should update");
        let revoked = store
            .get_voice(created.id.clone())
            .await
            .expect("voice lookup should succeed")
            .expect("voice should exist");
        assert!(!revoked.allows_local_tts_use());
        assert_eq!(revoked.reference_language.as_deref(), Some("hi"));

        assert!(store
            .delete_voice(created.id.clone())
            .await
            .expect("voice should delete"));
        assert!(store
            .get_voice(created.id)
            .await
            .expect("voice lookup should succeed")
            .is_none());

        clear_env();
    }
}

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;
use std::path::PathBuf;
use tokio::task;

use crate::storage_layout;

#[derive(Debug, Clone, Serialize)]
pub struct VoiceObservation {
    pub id: String,
    pub profile_id: String,
    pub category: String,
    pub summary: String,
    pub confidence: f32,
    pub source_turn_id: Option<String>,
    pub source_user_text: Option<String>,
    pub source_assistant_text: Option<String>,
    pub times_seen: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub forgotten_at: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CandidateObservation {
    pub category: String,
    pub summary: String,
    pub confidence: f32,
}

#[derive(Clone)]
pub struct VoiceObservationStore {
    db_path: PathBuf,
}

impl VoiceObservationStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare voice observation storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!(
                "Failed to open voice observation database: {}",
                db_path.display()
            )
        })?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS voice_observations (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                canonical_summary TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_turn_id TEXT NULL,
                source_user_text TEXT NULL,
                source_assistant_text TEXT NULL,
                times_seen INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                forgotten_at INTEGER NULL,
                FOREIGN KEY(profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_voice_observations_profile_updated_at
                ON voice_observations(profile_id, updated_at DESC, created_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_observations_profile_canonical_active
                ON voice_observations(profile_id, canonical_summary)
                WHERE forgotten_at IS NULL;
            "#,
        )
        .context("Failed to initialize voice observation database schema")?;

        Ok(Self { db_path })
    }

    pub async fn list_active(
        &self,
        profile_id: String,
        limit: usize,
    ) -> anyhow::Result<Vec<VoiceObservation>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    profile_id,
                    category,
                    summary,
                    confidence,
                    source_turn_id,
                    source_user_text,
                    source_assistant_text,
                    times_seen,
                    created_at,
                    updated_at,
                    forgotten_at
                FROM voice_observations
                WHERE profile_id = ?1
                  AND forgotten_at IS NULL
                ORDER BY confidence DESC, updated_at DESC, created_at DESC
                LIMIT ?2
                "#,
            )?;
            let rows = stmt.query_map(params![profile_id, limit.max(1) as i64], map_observation)?;
            let mut observations = Vec::new();
            for row in rows {
                observations.push(row?);
            }
            Ok(observations)
        })
        .await
    }

    pub async fn upsert_candidates(
        &self,
        profile_id: String,
        source_turn_id: Option<String>,
        source_user_text: Option<String>,
        source_assistant_text: Option<String>,
        candidates: Vec<CandidateObservation>,
    ) -> anyhow::Result<Vec<VoiceObservation>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let mut persisted = Vec::new();
            let now = now_unix_millis_i64();

            for candidate in candidates {
                let Some(summary) = sanitize_summary(candidate.summary.as_str()) else {
                    continue;
                };
                let category = sanitize_category(candidate.category.as_str());
                let confidence = clamp_confidence(candidate.confidence);
                let canonical_summary =
                    build_canonical_summary(category.as_str(), summary.as_str());

                let existing_id = tx
                    .query_row(
                        r#"
                        SELECT id
                        FROM voice_observations
                        WHERE profile_id = ?1
                          AND canonical_summary = ?2
                          AND forgotten_at IS NULL
                        LIMIT 1
                        "#,
                        params![profile_id, canonical_summary],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?;

                let observation = if let Some(existing_id) = existing_id {
                    tx.execute(
                        r#"
                        UPDATE voice_observations
                        SET confidence = MAX(confidence, ?1),
                            source_turn_id = ?2,
                            source_user_text = ?3,
                            source_assistant_text = ?4,
                            times_seen = times_seen + 1,
                            updated_at = ?5
                        WHERE id = ?6
                        "#,
                        params![
                            confidence,
                            sanitize_optional_text(source_turn_id.as_deref(), 160),
                            sanitize_optional_text(source_user_text.as_deref(), 16000),
                            sanitize_optional_text(source_assistant_text.as_deref(), 16000),
                            now,
                            existing_id,
                        ],
                    )?;
                    fetch_observation(&tx, &existing_id)?
                        .ok_or_else(|| anyhow!("Updated observation not found"))?
                } else {
                    let observation_id = format!("voice_obs_{}", uuid::Uuid::new_v4().simple());
                    tx.execute(
                        r#"
                        INSERT INTO voice_observations (
                            id,
                            profile_id,
                            category,
                            summary,
                            canonical_summary,
                            confidence,
                            source_turn_id,
                            source_user_text,
                            source_assistant_text,
                            times_seen,
                            created_at,
                            updated_at,
                            forgotten_at
                        )
                        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10, ?10, NULL)
                        "#,
                        params![
                            observation_id,
                            profile_id,
                            category,
                            summary,
                            canonical_summary,
                            confidence,
                            sanitize_optional_text(source_turn_id.as_deref(), 160),
                            sanitize_optional_text(source_user_text.as_deref(), 16000),
                            sanitize_optional_text(source_assistant_text.as_deref(), 16000),
                            now,
                        ],
                    )?;
                    fetch_observation(&tx, &observation_id)?
                        .ok_or_else(|| anyhow!("Inserted observation not found"))?
                };

                persisted.push(observation);
            }

            tx.commit()?;
            Ok(persisted)
        })
        .await
    }

    pub async fn forget_observation(&self, observation_id: String) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let updated = conn.execute(
                r#"
                UPDATE voice_observations
                SET forgotten_at = ?1,
                    updated_at = ?1
                WHERE id = ?2
                  AND forgotten_at IS NULL
                "#,
                params![now, observation_id],
            )?;
            Ok(updated > 0)
        })
        .await
    }

    pub async fn clear_profile(&self, profile_id: String) -> anyhow::Result<usize> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let updated = conn.execute(
                r#"
                UPDATE voice_observations
                SET forgotten_at = ?1,
                    updated_at = ?1
                WHERE profile_id = ?2
                  AND forgotten_at IS NULL
                "#,
                params![now, profile_id],
            )?;
            Ok(updated)
        })
        .await
    }

    pub async fn build_context(
        &self,
        profile_id: String,
        limit: usize,
    ) -> anyhow::Result<Option<String>> {
        let observations = self.list_active(profile_id, limit).await?;
        if observations.is_empty() {
            return Ok(None);
        }

        let mut lines = Vec::with_capacity(observations.len() + 2);
        lines.push(
            "Local observational memory about the user. Use it only if relevant, and do not mention the memory system.".to_string(),
        );
        for observation in observations {
            lines.push(format!(
                "- {}: {}",
                observation.category, observation.summary
            ));
        }
        Ok(Some(lines.join("\n")))
    }

    async fn run_blocking<F, T>(&self, task_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || task_fn(db_path))
            .await
            .map_err(|err| anyhow!("Voice observation storage worker failed: {err}"))?
    }
}

fn fetch_observation(
    conn: &Connection,
    observation_id: &str,
) -> anyhow::Result<Option<VoiceObservation>> {
    let observation = conn
        .query_row(
            r#"
            SELECT
                id,
                profile_id,
                category,
                summary,
                confidence,
                source_turn_id,
                source_user_text,
                source_assistant_text,
                times_seen,
                created_at,
                updated_at,
                forgotten_at
            FROM voice_observations
            WHERE id = ?1
            "#,
            params![observation_id],
            map_observation,
        )
        .optional()?;
    Ok(observation)
}

fn map_observation(row: &Row<'_>) -> rusqlite::Result<VoiceObservation> {
    let times_seen: i64 = row.get(8)?;
    Ok(VoiceObservation {
        id: row.get(0)?,
        profile_id: row.get(1)?,
        category: row.get(2)?,
        summary: row.get(3)?,
        confidence: row.get::<_, f64>(4)? as f32,
        source_turn_id: row.get(5)?,
        source_user_text: row.get(6)?,
        source_assistant_text: row.get(7)?,
        times_seen: times_seen.max(0) as usize,
        created_at: row.get::<_, i64>(9)?.max(0) as u64,
        updated_at: row.get::<_, i64>(10)?.max(0) as u64,
        forgotten_at: row
            .get::<_, Option<i64>>(11)?
            .map(|value| value.max(0) as u64),
    })
}

fn sanitize_category(raw: &str) -> String {
    let normalized = raw
        .trim()
        .to_lowercase()
        .replace('_', " ")
        .replace('-', " ");
    let simplified = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    if simplified.is_empty() {
        "general".to_string()
    } else {
        simplified.chars().take(48).collect()
    }
}

fn sanitize_summary(raw: &str) -> Option<String> {
    let normalized = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.chars().take(240).collect())
    }
}

fn sanitize_optional_text(raw: Option<&str>, max_len: usize) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(max_len).collect())
}

fn build_canonical_summary(category: &str, summary: &str) -> String {
    let mut normalized = String::with_capacity(category.len() + summary.len() + 1);
    normalized.push_str(category);
    normalized.push(':');
    for ch in summary.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
        } else if !normalized.ends_with(' ') {
            normalized.push(' ');
        }
    }
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clamp_confidence(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn now_unix_millis_i64() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_store::VoiceStore;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock")
    }

    #[tokio::test]
    async fn upserts_and_clears_observations() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("voice-observations.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let voice_store = VoiceStore::initialize().expect("voice store");
        let profile = voice_store.get_default_profile().await.expect("profile");
        let observation_store = VoiceObservationStore::initialize().expect("observation store");

        let inserted = observation_store
            .upsert_candidates(
                profile.id.clone(),
                Some("turn-1".to_string()),
                Some("Call me Lennex".to_string()),
                Some("Sure, Lennex.".to_string()),
                vec![CandidateObservation {
                    category: "preference".to_string(),
                    summary: "User prefers to be called Lennex".to_string(),
                    confidence: 0.92,
                }],
            )
            .await
            .expect("inserted");
        assert_eq!(inserted.len(), 1);

        let updated = observation_store
            .upsert_candidates(
                profile.id.clone(),
                Some("turn-2".to_string()),
                Some("Call me Lennex".to_string()),
                None,
                vec![CandidateObservation {
                    category: "preference".to_string(),
                    summary: "User prefers to be called Lennex".to_string(),
                    confidence: 0.75,
                }],
            )
            .await
            .expect("updated");
        assert_eq!(updated.len(), 1);

        let observations = observation_store
            .list_active(profile.id.clone(), 10)
            .await
            .expect("list");
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].times_seen, 2);

        let cleared = observation_store
            .clear_profile(profile.id.clone())
            .await
            .expect("clear");
        assert_eq!(cleared, 1);

        let empty = observation_store
            .list_active(profile.id, 10)
            .await
            .expect("list after clear");
        assert!(empty.is_empty());

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }
}

use anyhow::{anyhow, Context};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, EntityTrait, QueryFilter, QueryResult, TransactionTrait,
};
use serde::Serialize;

use crate::db::{raw, StoreDatabase};
use crate::entity::voice_observations;
use crate::ids::new_uuid;

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
    db: StoreDatabase,
}

impl VoiceObservationStore {
    pub fn initialize() -> anyhow::Result<Self> {
        Ok(Self {
            db: StoreDatabase::from_default_path()?,
        })
    }

    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self { db }
    }

    pub async fn list_active(
        &self,
        profile_id: String,
        limit: usize,
    ) -> anyhow::Result<Vec<VoiceObservation>> {
        let db = self.db.connection().await?;
        let limit = i64::try_from(limit.max(1)).context("Voice observation limit exceeds i64")?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                OBSERVATIONS_ACTIVE_SQL,
                vec![profile_id.into(), limit.into()],
            )?)
            .await
            .context("Failed to list voice observations")?;
        rows.iter().map(map_observation).collect()
    }

    pub async fn upsert_candidates(
        &self,
        profile_id: String,
        source_turn_id: Option<String>,
        source_user_text: Option<String>,
        source_assistant_text: Option<String>,
        candidates: Vec<CandidateObservation>,
    ) -> anyhow::Result<Vec<VoiceObservation>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start voice observation transaction")?;
        let mut persisted = Vec::new();
        let now = now_unix_millis_i64();

        for candidate in candidates {
            let Some(summary) = sanitize_summary(candidate.summary.as_str()) else {
                continue;
            };
            let category = sanitize_category(candidate.category.as_str());
            let confidence = clamp_confidence(candidate.confidence);
            let canonical_summary = build_canonical_summary(category.as_str(), summary.as_str());

            let existing_id = tx
                .query_one_raw(raw::statement(
                    &tx,
                    r#"
                    SELECT id
                    FROM voice_observations
                    WHERE profile_id = ?1
                      AND canonical_summary = ?2
                      AND forgotten_at IS NULL
                    LIMIT 1
                    "#,
                    vec![profile_id.clone().into(), canonical_summary.clone().into()],
                )?)
                .await
                .context("Failed to find existing voice observation")?
                .map(|row| row.try_get_by_index::<String>(0))
                .transpose()?;

            let observation = if let Some(existing_id) = existing_id {
                let confidence_expr = raw::greatest(tx.get_database_backend(), "confidence", "?1")?;
                tx.execute_raw(raw::statement(
                    &tx,
                    format!(
                        r#"
                    UPDATE voice_observations
                    SET confidence = {confidence_expr},
                        source_turn_id = ?2,
                        source_user_text = ?3,
                        source_assistant_text = ?4,
                        times_seen = times_seen + 1,
                        updated_at = ?5
                    WHERE id = ?6
                    "#
                    ),
                    vec![
                        f64::from(confidence).into(),
                        sanitize_optional_text(source_turn_id.as_deref(), 160).into(),
                        sanitize_optional_text(source_user_text.as_deref(), 16000).into(),
                        sanitize_optional_text(source_assistant_text.as_deref(), 16000).into(),
                        now.into(),
                        existing_id.clone().into(),
                    ],
                )?)
                .await
                .context("Failed to update voice observation")?;
                fetch_observation(&tx, &existing_id)
                    .await?
                    .ok_or_else(|| anyhow!("Updated observation not found"))?
            } else {
                let observation_id = new_uuid();
                tx.execute_raw(raw::statement(
                    &tx,
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
                    vec![
                        observation_id.clone().into(),
                        profile_id.clone().into(),
                        category.into(),
                        summary.into(),
                        canonical_summary.into(),
                        f64::from(confidence).into(),
                        sanitize_optional_text(source_turn_id.as_deref(), 160).into(),
                        sanitize_optional_text(source_user_text.as_deref(), 16000).into(),
                        sanitize_optional_text(source_assistant_text.as_deref(), 16000).into(),
                        now.into(),
                    ],
                )?)
                .await
                .context("Failed to insert voice observation")?;
                fetch_observation(&tx, &observation_id)
                    .await?
                    .ok_or_else(|| anyhow!("Inserted observation not found"))?
            };

            persisted.push(observation);
        }

        tx.commit()
            .await
            .context("Failed to commit voice observation transaction")?;
        Ok(persisted)
    }

    pub async fn forget_observation(&self, observation_id: String) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let result = voice_observations::Entity::update_many()
            .col_expr(voice_observations::Column::ForgottenAt, Expr::value(now))
            .col_expr(voice_observations::Column::UpdatedAt, Expr::value(now))
            .filter(voice_observations::Column::Id.eq(observation_id))
            .filter(voice_observations::Column::ForgottenAt.is_null())
            .exec(db)
            .await
            .context("Failed to forget voice observation")?;
        Ok(result.rows_affected > 0)
    }

    pub async fn clear_profile(&self, profile_id: String) -> anyhow::Result<usize> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let result = voice_observations::Entity::update_many()
            .col_expr(voice_observations::Column::ForgottenAt, Expr::value(now))
            .col_expr(voice_observations::Column::UpdatedAt, Expr::value(now))
            .filter(voice_observations::Column::ProfileId.eq(profile_id))
            .filter(voice_observations::Column::ForgottenAt.is_null())
            .exec(db)
            .await
            .context("Failed to clear voice observations")?;
        usize::try_from(result.rows_affected).context("Voice observation update count overflow")
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
}

const OBSERVATIONS_ACTIVE_SQL: &str = r#"
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
"#;

const OBSERVATION_BY_ID_SQL: &str = r#"
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
"#;

async fn fetch_observation<C>(
    conn: &C,
    observation_id: &str,
) -> anyhow::Result<Option<VoiceObservation>>
where
    C: ConnectionTrait,
{
    let row = conn
        .query_one_raw(raw::statement(
            conn,
            OBSERVATION_BY_ID_SQL,
            vec![observation_id.into()],
        )?)
        .await
        .context("Failed to load voice observation")?;
    row.as_ref().map(map_observation).transpose()
}

fn map_observation(row: &QueryResult) -> anyhow::Result<VoiceObservation> {
    let times_seen: i64 = row.try_get_by_index(8)?;
    Ok(VoiceObservation {
        id: row.try_get_by_index(0)?,
        profile_id: row.try_get_by_index(1)?,
        category: row.try_get_by_index(2)?,
        summary: row.try_get_by_index(3)?,
        confidence: row.try_get_by_index::<f64>(4)? as f32,
        source_turn_id: row.try_get_by_index(5)?,
        source_user_text: row.try_get_by_index(6)?,
        source_assistant_text: row.try_get_by_index(7)?,
        times_seen: times_seen.max(0) as usize,
        created_at: row.try_get_by_index::<i64>(9)?.max(0) as u64,
        updated_at: row.try_get_by_index::<i64>(10)?.max(0) as u64,
        forgotten_at: row
            .try_get_by_index::<Option<i64>>(11)?
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
    use crate::test_support::env_lock;
    use crate::voice_store::VoiceStore;

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

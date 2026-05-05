//! Persistent first-run onboarding state backed by SQLite.

use anyhow::Context;
use sea_orm::sea_query::OnConflict;
use sea_orm::{DatabaseConnection, EntityTrait, Set};
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::StoreDatabase;
use crate::entity::onboarding_state;

const ONBOARDING_STATE_ID: &str = "default";

#[derive(Debug, Clone, Serialize)]
pub struct OnboardingState {
    pub completed: bool,
    pub completed_at: Option<u64>,
    pub analytics_opt_in: bool,
}

#[derive(Clone)]
pub struct OnboardingStore {
    db: StoreDatabase,
}

impl OnboardingStore {
    pub fn initialize() -> anyhow::Result<Self> {
        Ok(Self {
            db: StoreDatabase::from_default_path()?,
        })
    }

    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self { db }
    }

    pub async fn get_state(&self) -> anyhow::Result<OnboardingState> {
        let db = self.db.connection().await?;
        read_state(db).await
    }

    pub async fn mark_completed(&self) -> anyhow::Result<OnboardingState> {
        let db = self.db.connection().await?;
        let completed_at = current_timestamp();
        let state = onboarding_state::ActiveModel {
            id: Set(ONBOARDING_STATE_ID.to_string()),
            completed_at: Set(Some(completed_at as i64)),
            ..Default::default()
        };

        onboarding_state::Entity::insert(state)
            .on_conflict(
                OnConflict::column(onboarding_state::Column::Id)
                    .update_column(onboarding_state::Column::CompletedAt)
                    .to_owned(),
            )
            .exec(db)
            .await
            .context("Failed to mark onboarding completed")?;

        read_state(db).await
    }

    pub async fn set_analytics_opt_in(
        &self,
        analytics_opt_in: bool,
    ) -> anyhow::Result<OnboardingState> {
        let db = self.db.connection().await?;
        let state = onboarding_state::ActiveModel {
            id: Set(ONBOARDING_STATE_ID.to_string()),
            analytics_opt_in: Set(bool_to_i64(analytics_opt_in)),
            ..Default::default()
        };

        onboarding_state::Entity::insert(state)
            .on_conflict(
                OnConflict::column(onboarding_state::Column::Id)
                    .update_column(onboarding_state::Column::AnalyticsOptIn)
                    .to_owned(),
            )
            .exec(db)
            .await
            .context("Failed to update onboarding analytics preference")?;

        read_state(db).await
    }
}

async fn read_state(db: &DatabaseConnection) -> anyhow::Result<OnboardingState> {
    let model = onboarding_state::Entity::find_by_id(ONBOARDING_STATE_ID.to_string())
        .one(db)
        .await
        .context("Failed to load onboarding state")?;
    Ok(model_to_state(model))
}

fn model_to_state(model: Option<onboarding_state::Model>) -> OnboardingState {
    let Some(model) = model else {
        return OnboardingState {
            completed: false,
            completed_at: None,
            analytics_opt_in: false,
        };
    };

    let completed_at = model.completed_at.and_then(i64_to_u64);
    OnboardingState {
        completed: completed_at.is_some(),
        completed_at,
        analytics_opt_in: i64_to_bool(model.analytics_opt_in),
    }
}

fn i64_to_u64(value: i64) -> Option<u64> {
    if value > 0 { Some(value as u64) } else { None }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn bool_to_i64(value: bool) -> i64 {
    if value { 1 } else { 0 }
}

fn i64_to_bool(value: i64) -> bool {
    value > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use std::future::Future;
    use tempfile::TempDir;

    async fn with_env_lock<T>(action: impl Future<Output = T>) -> T {
        let _guard = env_lock();
        action.await
    }

    fn setup_store() -> (TempDir, OnboardingStore) {
        let temp_dir = tempfile::tempdir().expect("temp dir should create");
        let db_path = temp_dir.path().join("izwi.sqlite3");
        let media_root = temp_dir.path().join("media");

        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_root);

        let store = OnboardingStore::initialize().expect("store should init");
        (temp_dir, store)
    }

    fn clear_env() {
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    #[tokio::test]
    async fn onboarding_defaults_to_incomplete() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let state = store.get_state().await.expect("state should load");
            assert!(!state.completed);
            assert!(state.completed_at.is_none());
            assert!(!state.analytics_opt_in);
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn mark_completed_sets_timestamp() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let state = store.mark_completed().await.expect("state should update");
            assert!(state.completed);
            assert!(state.completed_at.is_some());
            assert!(!state.analytics_opt_in);

            let reload = store.get_state().await.expect("state should reload");
            assert!(reload.completed);
            assert!(reload.completed_at.is_some());
            assert!(!reload.analytics_opt_in);
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn analytics_opt_in_updates_without_losing_completion_state() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let initial = store.get_state().await.expect("state should load");
            assert!(!initial.analytics_opt_in);

            let opted_in = store
                .set_analytics_opt_in(true)
                .await
                .expect("opt-in should persist");
            assert!(opted_in.analytics_opt_in);
            assert!(!opted_in.completed);
            assert!(opted_in.completed_at.is_none());

            let completed = store.mark_completed().await.expect("state should complete");
            assert!(completed.analytics_opt_in);
            assert!(completed.completed);
            assert!(completed.completed_at.is_some());

            let opted_out = store
                .set_analytics_opt_in(false)
                .await
                .expect("opt-out should persist");
            assert!(!opted_out.analytics_opt_in);
            assert!(opted_out.completed);
            assert!(opted_out.completed_at.is_some());
            clear_env();
        })
        .await;
    }
}

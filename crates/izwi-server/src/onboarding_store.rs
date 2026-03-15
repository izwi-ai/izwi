//! Persistent first-run onboarding state backed by SQLite.

use anyhow::Context;
use rusqlite::{params, OptionalExtension};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout;

const ONBOARDING_STATE_ID: &str = "default";

#[derive(Debug, Clone, Serialize)]
pub struct OnboardingState {
    pub completed: bool,
    pub completed_at: Option<u64>,
}

#[derive(Clone)]
pub struct OnboardingStore {
    db_path: PathBuf,
}

impl OnboardingStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare onboarding storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path)
            .with_context(|| format!("Failed to open onboarding database: {}", db_path.display()))?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS onboarding_state (
                id TEXT PRIMARY KEY,
                completed_at INTEGER NULL
            );
            "#,
        )
        .context("Failed to initialize onboarding database schema")?;

        Ok(Self { db_path })
    }

    pub async fn get_state(&self) -> anyhow::Result<OnboardingState> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let completed_at: Option<i64> = conn
                .query_row(
                    r#"
                    SELECT completed_at
                    FROM onboarding_state
                    WHERE id = ?1
                    "#,
                    params![ONBOARDING_STATE_ID],
                    |row| row.get(0),
                )
                .optional()?;

            Ok(to_state(completed_at))
        })
        .await
    }

    pub async fn mark_completed(&self) -> anyhow::Result<OnboardingState> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let completed_at = current_timestamp();

            conn.execute(
                r#"
                INSERT INTO onboarding_state (id, completed_at)
                VALUES (?1, ?2)
                ON CONFLICT(id) DO UPDATE SET completed_at = excluded.completed_at
                "#,
                params![ONBOARDING_STATE_ID, completed_at as i64],
            )?;

            Ok(OnboardingState {
                completed: true,
                completed_at: Some(completed_at),
            })
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
            .context("Onboarding store task failed")?
    }
}

fn to_state(completed_at: Option<i64>) -> OnboardingState {
    let completed_at = completed_at.and_then(i64_to_u64);
    OnboardingState {
        completed: completed_at.is_some(),
        completed_at,
    }
}

fn i64_to_u64(value: i64) -> Option<u64> {
    if value > 0 {
        Some(value as u64)
    } else {
        None
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::OnceLock;
    use tokio::sync::Mutex;
    use tempfile::TempDir;

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    async fn with_env_lock<T>(action: impl Future<Output = T>) -> T {
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().await;
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

            let reload = store.get_state().await.expect("state should reload");
            assert!(reload.completed);
            assert!(reload.completed_at.is_some());
            clear_env();
        })
        .await;
    }
}

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
    pub analytics_opt_in: bool,
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

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!("Failed to open onboarding database: {}", db_path.display())
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS onboarding_state (
                id TEXT PRIMARY KEY,
                completed_at INTEGER NULL,
                analytics_opt_in INTEGER NOT NULL DEFAULT 0
            );
            "#,
        )
        .context("Failed to initialize onboarding database schema")?;
        ensure_onboarding_state_analytics_opt_in_column(&conn)?;

        Ok(Self { db_path })
    }

    pub async fn get_state(&self) -> anyhow::Result<OnboardingState> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            read_state_from_conn(&conn)
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

            read_state_from_conn(&conn)
        })
        .await
    }

    pub async fn set_analytics_opt_in(
        &self,
        analytics_opt_in: bool,
    ) -> anyhow::Result<OnboardingState> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let analytics_opt_in_int = bool_to_i64(analytics_opt_in);

            conn.execute(
                r#"
                INSERT INTO onboarding_state (id, completed_at, analytics_opt_in)
                VALUES (?1, NULL, ?2)
                ON CONFLICT(id) DO UPDATE SET analytics_opt_in = excluded.analytics_opt_in
                "#,
                params![ONBOARDING_STATE_ID, analytics_opt_in_int],
            )?;

            read_state_from_conn(&conn)
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

fn read_state_from_conn(conn: &rusqlite::Connection) -> anyhow::Result<OnboardingState> {
    let row = conn
        .query_row(
            r#"
            SELECT completed_at, analytics_opt_in
            FROM onboarding_state
            WHERE id = ?1
            "#,
            params![ONBOARDING_STATE_ID],
            |row| {
                let completed_at: Option<i64> = row.get(0)?;
                let analytics_opt_in: i64 = row.get(1)?;
                Ok((completed_at, analytics_opt_in))
            },
        )
        .optional()?;

    let (completed_at_raw, analytics_opt_in_raw) = row.unwrap_or((None, 0));
    let completed_at = completed_at_raw.and_then(i64_to_u64);
    Ok(OnboardingState {
        completed: completed_at.is_some(),
        completed_at,
        analytics_opt_in: i64_to_bool(analytics_opt_in_raw),
    })
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

fn bool_to_i64(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

fn i64_to_bool(value: i64) -> bool {
    value > 0
}

fn ensure_onboarding_state_analytics_opt_in_column(
    conn: &rusqlite::Connection,
) -> anyhow::Result<()> {
    if onboarding_state_has_column(conn, "analytics_opt_in")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE onboarding_state ADD COLUMN analytics_opt_in INTEGER NOT NULL DEFAULT 0",
        [],
    )
    .context("Failed adding onboarding_state.analytics_opt_in column")?;

    Ok(())
}

fn onboarding_state_has_column(conn: &rusqlite::Connection, target: &str) -> anyhow::Result<bool> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(onboarding_state)")
        .context("Failed to inspect onboarding_state schema")?;
    let mut rows = stmt
        .query([])
        .context("Failed to query onboarding_state schema info")?;

    while let Some(row) = rows
        .next()
        .context("Failed reading onboarding_state schema row")?
    {
        let name: String = row
            .get(1)
            .context("Failed reading onboarding_state column name")?;
        if name == target {
            return Ok(true);
        }
    }

    Ok(false)
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

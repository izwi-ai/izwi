use crate::{db::migrator::Migrator, storage_layout};
use anyhow::Context;
use sea_orm::{
    sqlx::sqlite::SqliteJournalMode, ConnectOptions, Database, DatabaseConnection,
};
use std::path::Path;
use std::time::Duration;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(3);
const SQLITE_MAX_CONNECTIONS: u32 = 5;

pub async fn initialize_default() -> anyhow::Result<DatabaseConnection> {
    let db = connect_default().await?;
    Migrator::up(&db)
        .await
        .context("Failed to run SQLite migrations")?;
    Ok(db)
}

pub async fn connect_default() -> anyhow::Result<DatabaseConnection> {
    let db_path = storage_layout::resolve_db_path();
    let media_root = storage_layout::resolve_media_root();
    storage_layout::ensure_storage_dirs(&db_path, &media_root)
        .context("Failed to prepare storage layout for SeaORM")?;

    connect_path(&db_path).await
}

pub async fn connect_path(db_path: &Path) -> anyhow::Result<DatabaseConnection> {
    let mut options = ConnectOptions::new(sqlite_url(db_path));
    options
        .max_connections(SQLITE_MAX_CONNECTIONS)
        .min_connections(1)
        .connect_timeout(SQLITE_BUSY_TIMEOUT)
        .acquire_timeout(SQLITE_BUSY_TIMEOUT)
        .sqlx_logging(false)
        .map_sqlx_sqlite_opts(|options| {
            options
                .create_if_missing(true)
                .busy_timeout(SQLITE_BUSY_TIMEOUT)
                .foreign_keys(true)
                .journal_mode(SqliteJournalMode::Wal)
        });

    Database::connect(options)
        .await
        .with_context(|| format!("Unable to open SeaORM SQLite database at {}", db_path.display()))
}

fn sqlite_url(db_path: &Path) -> String {
    let path = db_path.to_string_lossy();
    let escaped = path
        .replace('%', "%25")
        .replace('?', "%3F")
        .replace('#', "%23");
    format!("sqlite://{escaped}?mode=rwc")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use sea_orm::{ConnectionTrait, DbBackend, Statement};

    #[tokio::test]
    async fn default_connection_preserves_sqlite_pragmas() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("seaorm.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let db = initialize_default().await.expect("database initializes");
        assert_eq!(db.get_database_backend(), DbBackend::Sqlite);

        let foreign_keys = query_i64_pragma(&db, "PRAGMA foreign_keys")
            .await
            .expect("foreign_keys pragma");
        assert_eq!(foreign_keys, 1);

        let journal_mode = query_string_pragma(&db, "PRAGMA journal_mode")
            .await
            .expect("journal_mode pragma");
        assert_eq!(journal_mode.to_ascii_lowercase(), "wal");

        assert!(db_path.exists());
        assert!(media_dir.exists());

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    async fn query_i64_pragma(db: &DatabaseConnection, sql: &str) -> anyhow::Result<i64> {
        let row = db
            .query_one_raw(Statement::from_string(DbBackend::Sqlite, sql.to_string()))
            .await?
            .context("pragma returned no row")?;
        Ok(row.try_get_by_index(0)?)
    }

    async fn query_string_pragma(db: &DatabaseConnection, sql: &str) -> anyhow::Result<String> {
        let row = db
            .query_one_raw(Statement::from_string(DbBackend::Sqlite, sql.to_string()))
            .await?
            .context("pragma returned no row")?;
        Ok(row.try_get_by_index(0)?)
    }
}

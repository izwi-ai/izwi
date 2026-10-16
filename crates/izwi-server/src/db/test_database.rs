//! Disposable databases for store tests.
//!
//! A test database is a SQLite file in a temporary directory. When
//! `IZWI_TEST_STORE_PG_URL` names a disposable PostgreSQL database (and the
//! `db-postgres` feature is on), it is instead a fresh schema in that
//! database, so the same tests run against PostgreSQL. Schemas are left
//! behind for inspection; the database must not hold anything else.

use super::StoreDatabase;
use std::path::PathBuf;
use tempfile::TempDir;

pub(crate) const STORE_PG_URL_ENV: &str = "IZWI_TEST_STORE_PG_URL";

#[derive(Debug, Clone)]
enum Location {
    Sqlite(PathBuf),
    #[cfg(feature = "db-postgres")]
    Postgres(String),
}

pub(crate) struct TestDatabase {
    location: Location,
    _root: TempDir,
}

impl TestDatabase {
    pub(crate) async fn new() -> Self {
        let root = tempfile::tempdir().expect("temp dir");
        let location = match postgres_url() {
            #[cfg(feature = "db-postgres")]
            Some(url) => Location::Postgres(fresh_schema(&url).await),
            #[cfg(not(feature = "db-postgres"))]
            Some(_) => panic!("{STORE_PG_URL_ENV} needs izwi-server's db-postgres feature"),
            None => Location::Sqlite(root.path().join("runtime.sqlite")),
        };
        Self {
            location,
            _root: root,
        }
    }

    /// A new handle, with its own pool, on this test's database.
    pub(crate) fn database(&self) -> StoreDatabase {
        match &self.location {
            Location::Sqlite(path) => StoreDatabase::new(path.clone()),
            #[cfg(feature = "db-postgres")]
            Location::Postgres(url) => StoreDatabase::from_url(url.clone()),
        }
    }
}

fn postgres_url() -> Option<String> {
    std::env::var(STORE_PG_URL_ENV)
        .ok()
        .filter(|url| !url.trim().is_empty())
}

/// Create an empty schema and address it through the connection's
/// `search_path`, so every unqualified table lives in it.
#[cfg(feature = "db-postgres")]
async fn fresh_schema(url: &str) -> String {
    use sea_orm::ConnectionTrait;

    let schema = format!("izwi_test_{}", uuid::Uuid::new_v4().simple());
    let admin = sea_orm::Database::connect(url)
        .await
        .expect("test PostgreSQL database opens");
    admin
        .execute_unprepared(&format!("CREATE SCHEMA {schema}"))
        .await
        .expect("test schema is created");
    let _ = admin.close().await;
    let separator = if url.contains('?') { '&' } else { '?' };
    format!("{url}{separator}options[search_path]={schema}")
}

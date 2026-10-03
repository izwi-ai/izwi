pub mod fleet;
pub mod speech_progress;
pub mod store;
pub mod types;
pub mod worker;

// DS5.1 execution evidence against real PostgreSQL; compiled only when the
// server database backends are enabled so the default build stays hermetic.
#[cfg(all(test, feature = "db-postgres"))]
mod fleet_postgres_execution;

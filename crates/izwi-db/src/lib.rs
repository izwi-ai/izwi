//! Izwi's SeaORM-backed durable store: connection setup for SQLite and
//! PostgreSQL, the application and fleet-coordination schema migrations,
//! raw statement helpers and the on-disk storage layout.
//!
//! izwi-server's stores and the gateway's fleet coordination and principal
//! keys share it. It never depends on izwi-core.

// Async tests hold a process-wide environment lock across awaits so parallel
// tests cannot observe transient environment overrides (the policy these
// modules had in izwi-server). Production code keeps the stricter lint.
#![cfg_attr(test, allow(clippy::await_holding_lock))]

pub mod migrator;
pub mod raw;
pub mod schema_contract;
pub mod sqlite;
pub mod storage_layout;
#[cfg(any(test, feature = "test-support"))]
pub mod test_database;
#[cfg(test)]
mod test_support;
pub mod voice_defaults;

use sea_orm::{SqliteTransactionMode, TransactionOptions};

pub use sqlite::StoreDatabase;

/// Options for a transaction that writes durable state. SQLite must acquire
/// its write reservation before reading a snapshot: DEFERRED promotion fails
/// immediately with SQLITE_BUSY_SNAPSHOT (which the busy timeout does not
/// retry) whenever another connection commits between the transaction's first
/// read and first write. Other backends ignore the SQLite option.
pub fn write_transaction_options() -> TransactionOptions {
    TransactionOptions {
        sqlite_transaction_mode: Some(SqliteTransactionMode::Immediate),
        ..TransactionOptions::default()
    }
}

/// The current time in Unix milliseconds by this process's wall clock.
pub fn current_timestamp_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

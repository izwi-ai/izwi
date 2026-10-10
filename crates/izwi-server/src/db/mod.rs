//! SeaORM-backed database initialization.

pub mod migrator;
pub mod raw;
pub mod schema_contract;
pub mod sqlite;

use sea_orm::{SqliteTransactionMode, TransactionOptions};

pub use sqlite::StoreDatabase;

/// Options for a transaction that writes durable state. SQLite must acquire
/// its write reservation before reading a snapshot: DEFERRED promotion fails
/// immediately with SQLITE_BUSY_SNAPSHOT (which the busy timeout does not
/// retry) whenever another connection commits between the transaction's first
/// read and first write. Other backends ignore the SQLite option.
pub(crate) fn write_transaction_options() -> TransactionOptions {
    TransactionOptions {
        sqlite_transaction_mode: Some(SqliteTransactionMode::Immediate),
        ..TransactionOptions::default()
    }
}

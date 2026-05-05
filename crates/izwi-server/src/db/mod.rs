//! SeaORM-backed database initialization.

pub mod migrator;
pub mod sqlite;

pub use sqlite::StoreDatabase;

//! Backend-neutral contracts for communication between Izwi gateways and workers.
//!
//! This crate deliberately contains no inference-engine or accelerator dependencies. It defines
//! the private HTTP wire schema and bounded incremental NDJSON decoding used at the process
//! boundary. Public API compatibility remains the gateway's responsibility.

mod identity;
mod ndjson;
mod types;

pub use identity::*;
pub use ndjson::*;
pub use types::*;

/// The only protocol major version implemented by this crate.
pub const PROTOCOL_MAJOR_VERSION: u16 = 1;
/// The latest additive protocol minor version implemented by this crate.
pub const PROTOCOL_MINOR_VERSION: u16 = 0;
pub const PROTOCOL_V1: SchemaVersion =
    SchemaVersion::new(PROTOCOL_MAJOR_VERSION, PROTOCOL_MINOR_VERSION);

pub const WORKER_DESCRIPTOR_PATH: &str = "/internal/v1/worker";
pub const WORKER_STATUS_PATH: &str = "/internal/v1/status";
pub const INVOCATIONS_PATH: &str = "/internal/v1/invocations";
pub const SERVICE_AUTHORIZATION_HEADER: &str = "authorization";
pub const SERVICE_CREDENTIAL_ID_HEADER: &str = "x-izwi-service-credential-id";
pub const SERVICE_AUTH_SCHEME: &str = "Bearer";
pub const NDJSON_MEDIA_TYPE: &str = "application/x-ndjson";

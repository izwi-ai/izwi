//! Backend-neutral contracts for communication between Izwi gateways and workers.
//!
//! This crate deliberately contains no inference-engine or accelerator dependencies. It defines
//! the private HTTP wire schema, the bounded incremental NDJSON decoding used at the process
//! boundary, and the versioned realtime WebSocket subprotocol. Public API compatibility remains
//! the gateway's responsibility.

mod approvals;
mod identity;
mod ndjson;
mod realtime;
pub mod secret_file;
pub mod trace;
mod types;

pub use approvals::*;
pub use identity::*;
/// The shared environment-variable policy (re-exported for the serving crates).
pub use izwi_env as env;
pub use ndjson::*;
pub use realtime::*;
pub use trace::{TraceContext, TRACEPARENT_HEADER};
pub use types::*;

/// The only protocol major version implemented by this crate.
pub const PROTOCOL_MAJOR_VERSION: u16 = 1;
/// The latest additive protocol minor version implemented by this crate.
///
/// Minor 1 adds optional per-deployment routing signals on `LoadedDeployment`.
/// Minor 2 adds the realtime WebSocket subprotocol (`realtime` module), the
/// `WorkerFeature::RealtimeSocket` and `InputFormat::Text` variants. Minor 3
/// adds optional per-request cache accounting: `Usage::cached_tokens` (DS9.1)
/// and, later in minor 3, optional per-token logprobs on `TextDelta` (DS9.3).
/// Minor 4 adds the full chat control set on `ChatParameters` (top-k,
/// penalties, logprobs, JSON mode and the D8 context policy), one
/// `WorkerFeature` per control, `RejectionCode::ContextLengthExceeded`, and
/// version negotiation: every enum a worker sends tolerates later variants
/// (`Unknown`), a gateway sends its version in [`PROTOCOL_VERSION_HEADER`]
/// so a worker leaves out what the gateway cannot decode, and an invocation
/// carries the version negotiated from the worker's descriptor.
/// All version gates tolerate any minor within major 1: older workers omit
/// the fields and newer workers only add optional ones.
pub const PROTOCOL_MINOR_VERSION: u16 = 4;
pub const PROTOCOL_V1: SchemaVersion =
    SchemaVersion::new(PROTOCOL_MAJOR_VERSION, PROTOCOL_MINOR_VERSION);

pub const WORKER_DESCRIPTOR_PATH: &str = "/internal/v1/worker";
pub const WORKER_STATUS_PATH: &str = "/internal/v1/status";
pub const INVOCATIONS_PATH: &str = "/internal/v1/invocations";
/// Request header carrying the sender's `major.minor` protocol version.
pub const PROTOCOL_VERSION_HEADER: &str = "x-izwi-protocol-version";
/// The version assumed for a peer that sends no [`PROTOCOL_VERSION_HEADER`]:
/// the last minor before the header existed.
pub const UNDECLARED_PEER_VERSION: SchemaVersion = SchemaVersion::new(PROTOCOL_MAJOR_VERSION, 3);
pub const SERVICE_AUTHORIZATION_HEADER: &str = "authorization";
pub const SERVICE_CREDENTIAL_ID_HEADER: &str = "x-izwi-service-credential-id";
pub const SERVICE_AUTH_SCHEME: &str = "Bearer";
pub const NDJSON_MEDIA_TYPE: &str = "application/x-ndjson";

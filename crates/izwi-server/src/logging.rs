//! Process logging. The format, filter and subscriber are izwi-gateway's,
//! which both roles share; this server logs as `izwi-server`.

pub use izwi_gateway::logging::{LogFormat, SERVICE_VERSION};

pub const SERVICE_NAME: &str = "izwi-server";

pub fn init_tracing(log_format: LogFormat) {
    izwi_gateway::logging::init_tracing(log_format, SERVICE_NAME);
}

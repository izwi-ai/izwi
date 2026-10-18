//! Process logging: the text or JSON log format, the default filter and the
//! service name every log line and exported span carries.

use std::sync::OnceLock;

use clap::ValueEnum;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// The service name of the standalone gateway binary.
pub const GATEWAY_SERVICE_NAME: &str = "izwi-gateway";
pub const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The service name [`init_tracing`] set for this process.
static SERVICE_NAME: OnceLock<&'static str> = OnceLock::new();

/// The service this process logs as: the binary's name, as given to
/// [`init_tracing`] (`izwi-server` when the server runs the gateway role),
/// or [`GATEWAY_SERVICE_NAME`] before logging is set up.
pub fn service_name() -> &'static str {
    SERVICE_NAME.get().copied().unwrap_or(GATEWAY_SERVICE_NAME)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

impl LogFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

/// The log filter when `RUST_LOG` is unset: warnings from every Izwi crate
/// whose code runs in this process, and from the HTTP layer.
const DEFAULT_LOG_FILTER: &str =
    "izwi_server=warn,izwi_gateway=warn,izwi_db=warn,izwi_core=warn,tower_http=warn";

/// Install the process's log subscriber, logging as `service`.
pub fn init_tracing(log_format: LogFormat, service: &'static str) {
    let service = *SERVICE_NAME.get_or_init(|| service);
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| DEFAULT_LOG_FILTER.into());

    // Request spans go to an OTLP collector only when one is configured.
    let otlp = izwi_serving_client::otlp::otlp_layer_from_env(service);
    match log_format {
        LogFormat::Text => tracing_subscriber::registry()
            .with(filter)
            .with(otlp)
            .with(tracing_subscriber::fmt::layer())
            .init(),
        LogFormat::Json => tracing_subscriber::registry()
            .with(filter)
            .with(otlp)
            .with(json_layer(std::io::stdout))
            .init(),
    }
}

/// The JSON log line shape: event fields at the top level, plus the current
/// span and the span list, so request ids and W3C trace ids reach every line
/// logged inside a request.
pub fn json_layer<S, W>(writer: W) -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
    W: for<'writer> tracing_subscriber::fmt::MakeWriter<'writer> + Send + Sync + 'static,
{
    tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(writer)
}

#[cfg(test)]
mod tests {
    use super::DEFAULT_LOG_FILTER;
    use tracing::Level;
    use tracing_subscriber::filter::Targets;

    /// Code moved between crates keeps its warnings: each crate that runs in
    /// the server process logs at warn, and nothing more, by default.
    #[test]
    fn default_filter_keeps_every_crates_warnings() {
        let filter: Targets = DEFAULT_LOG_FILTER.parse().expect("valid directives");
        for target in [
            "izwi_server::roles::gateway",
            "izwi_gateway::gateway_worker_tls",
            "izwi_db::sqlite",
            "izwi_core::runtime",
        ] {
            assert!(filter.would_enable(target, &Level::WARN), "{target}");
            assert!(!filter.would_enable(target, &Level::INFO), "{target}");
        }
    }
}

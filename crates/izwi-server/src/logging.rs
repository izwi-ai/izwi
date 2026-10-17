use clap::ValueEnum;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub const SERVICE_NAME: &str = "izwi-server";
pub const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

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

pub fn init_tracing(log_format: LogFormat) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "izwi_server=warn,izwi_db=warn,izwi_core=warn,tower_http=warn".into());

    // Request spans go to an OTLP collector only when one is configured.
    let otlp = izwi_serving_client::otlp::otlp_layer_from_env(SERVICE_NAME);
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
pub(crate) fn json_layer<S, W>(writer: W) -> impl tracing_subscriber::Layer<S>
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

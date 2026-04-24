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
        .unwrap_or_else(|_| "izwi_server=warn,izwi_core=warn,tower_http=warn".into());

    match log_format {
        LogFormat::Text => tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init(),
        LogFormat::Json => tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json().flatten_event(true))
            .init(),
    }
}

//! What the gateway role runs from: its flags, its listener settings (bind
//! address, request deadline, CORS) and the settings it shares with the
//! local role's command line. `izwi-server --role gateway` derives them from
//! its own runtime configuration; the standalone gateway resolves them from
//! the same flags and environment variables.

use anyhow::{bail, Context};

use crate::api::openai::compat::OpenAiCompatibilityProfile;
use crate::cli::{BackendArg, GatewayArgs, GatewayCli};

const TIMEOUT_ENV: &str = "IZWI_TIMEOUT";
const LEGACY_TIMEOUT_ENV: &str = "REQUEST_TIMEOUT_SECS";
const CORS_ENV: &str = "IZWI_CORS";
const CORS_ORIGINS_ENV: &str = "IZWI_CORS_ORIGINS";
const COMPAT_PROFILE_ENV: &str = "IZWI_OPENAI_COMPAT_PROFILE";

/// Where and how the gateway's HTTP listener serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayServeConfig {
    /// Bind host (`--host`, `IZWI_HOST`).
    pub host: String,
    /// Bind port (`--port`, `IZWI_PORT`).
    pub port: u16,
    /// Deadline of one request, in seconds (`IZWI_TIMEOUT`).
    pub request_timeout_secs: u64,
    /// Whether browsers may call the gateway cross-origin (`IZWI_CORS`).
    pub cors_enabled: bool,
    /// The origins CORS allows (`IZWI_CORS_ORIGINS`).
    pub cors_origins: Vec<String>,
}

impl Default for GatewayServeConfig {
    /// The local server's defaults: loopback, port 8080, a 300 s request
    /// deadline and CORS off.
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            request_timeout_secs: 300,
            cors_enabled: false,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// Everything the gateway role runs from.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub args: GatewayArgs,
    pub serve: GatewayServeConfig,
    /// `--backend`: the worker backends chat may route to; `None` routes to
    /// any.
    pub backend: Option<BackendArg>,
    /// The OpenAI compatibility profile (`IZWI_OPENAI_COMPAT_PROFILE`).
    pub compat_profile: OpenAiCompatibilityProfile,
}

impl GatewayConfig {
    /// The standalone gateway's configuration: its command line, then the
    /// environment, then the defaults. A value that does not parse is an
    /// error naming its variable.
    pub fn from_cli(cli: GatewayCli) -> anyhow::Result<Self> {
        Self::from_cli_with_env(cli, |name| std::env::var(name).ok())
    }

    fn from_cli_with_env(
        cli: GatewayCli,
        env: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        let set = |name: &str| {
            env(name)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        let defaults = GatewayServeConfig::default();
        let request_timeout_secs = match set(TIMEOUT_ENV).or_else(|| set(LEGACY_TIMEOUT_ENV)) {
            Some(value) => value
                .parse::<u64>()
                .with_context(|| format!("{TIMEOUT_ENV} must be a whole number of seconds"))?,
            None => defaults.request_timeout_secs,
        };
        let cors_enabled = match set(CORS_ENV) {
            Some(value) => izwi_env::parse_bool(CORS_ENV, &value)?,
            None => defaults.cors_enabled,
        };
        let cors_origins = match set(CORS_ORIGINS_ENV) {
            Some(value) => value
                .split(',')
                .map(str::trim)
                .filter(|origin| !origin.is_empty())
                .map(ToString::to_string)
                .collect(),
            None => defaults.cors_origins,
        };
        Ok(Self {
            args: cli.gateway,
            serve: GatewayServeConfig {
                host: cli.host.unwrap_or(defaults.host),
                port: cli.port.unwrap_or(defaults.port),
                request_timeout_secs,
                cors_enabled,
                cors_origins,
            },
            backend: cli.backend,
            compat_profile: parse_compat_profile(set(COMPAT_PROFILE_ENV).as_deref())?,
        })
    }
}

/// `IZWI_OPENAI_COMPAT_PROFILE`: `strict` (the default) or `relaxed`.
pub fn parse_compat_profile(value: Option<&str>) -> anyhow::Result<OpenAiCompatibilityProfile> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None | Some("strict") => Ok(OpenAiCompatibilityProfile::Strict),
        Some("relaxed") => Ok(OpenAiCompatibilityProfile::Relaxed),
        Some(other) => {
            bail!("{COMPAT_PROFILE_ENV} must be strict or relaxed, not {other:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::collections::HashMap;

    fn resolve(args: &[&str], env: &[(&str, &str)]) -> anyhow::Result<GatewayConfig> {
        let env: HashMap<String, String> = env
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let cli = GatewayCli::try_parse_from(args)?;
        GatewayConfig::from_cli_with_env(cli, |name| env.get(name).cloned())
    }

    /// Unset, the standalone gateway serves like the server's gateway role
    /// with its defaults.
    #[test]
    fn defaults_match_the_server_gateway_role() {
        let config = resolve(&["izwi-gateway"], &[]).unwrap();
        assert_eq!(config.serve, GatewayServeConfig::default());
        assert!(config.backend.is_none());
        assert_eq!(config.compat_profile, OpenAiCompatibilityProfile::Strict);
        assert_eq!(config.args.gateway_max_in_flight, 32);
    }

    #[test]
    fn listener_settings_come_from_flags_then_the_environment() {
        let config = resolve(
            &["izwi-gateway", "--port", "9100", "--backend", "cuda"],
            &[
                ("IZWI_TIMEOUT", "30"),
                ("IZWI_CORS", "on"),
                ("IZWI_CORS_ORIGINS", "https://a.example, https://b.example"),
                ("IZWI_OPENAI_COMPAT_PROFILE", "relaxed"),
            ],
        )
        .unwrap();
        assert_eq!(config.serve.port, 9100);
        assert_eq!(config.serve.request_timeout_secs, 30);
        assert!(config.serve.cors_enabled);
        assert_eq!(
            config.serve.cors_origins,
            vec!["https://a.example", "https://b.example"]
        );
        assert!(matches!(config.backend, Some(BackendArg::Cuda)));
        assert_eq!(config.compat_profile, OpenAiCompatibilityProfile::Relaxed);

        let legacy = resolve(&["izwi-gateway"], &[("REQUEST_TIMEOUT_SECS", "45")]).unwrap();
        assert_eq!(legacy.serve.request_timeout_secs, 45);
    }

    /// The standalone gateway fails closed on a value it cannot read.
    #[test]
    fn unreadable_values_are_errors_naming_the_variable() {
        for (name, value) in [
            ("IZWI_TIMEOUT", "soon"),
            ("IZWI_CORS", "maybe"),
            ("IZWI_OPENAI_COMPAT_PROFILE", "loose"),
        ] {
            let error = resolve(&["izwi-gateway"], &[(name, value)]).unwrap_err();
            assert!(format!("{error:#}").contains(name), "{name}: {error:#}");
        }
    }
}

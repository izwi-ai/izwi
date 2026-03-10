use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::backends::BackendPreference;

pub const ENV_HOST: &str = "IZWI_HOST";
pub const ENV_PORT: &str = "IZWI_PORT";
pub const ENV_MODELS_DIR: &str = "IZWI_MODELS_DIR";
pub const ENV_BACKEND: &str = "IZWI_BACKEND";
pub const ENV_MAX_BATCH_SIZE: &str = "IZWI_MAX_BATCH_SIZE";
pub const ENV_NUM_THREADS: &str = "IZWI_NUM_THREADS";
pub const ENV_MAX_CONCURRENT: &str = "IZWI_MAX_CONCURRENT";
pub const ENV_TIMEOUT: &str = "IZWI_TIMEOUT";
pub const ENV_CORS: &str = "IZWI_CORS";
pub const ENV_CORS_ORIGINS: &str = "IZWI_CORS_ORIGINS";
pub const ENV_NO_UI: &str = "IZWI_NO_UI";
pub const ENV_UI_DIR: &str = "IZWI_UI_DIR";

pub const LEGACY_ENV_MAX_CONCURRENT: &[&str] = &["MAX_CONCURRENT_REQUESTS"];
pub const LEGACY_ENV_TIMEOUT: &[&str] = &["REQUEST_TIMEOUT_SECS"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeRuntimeConfig {
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
    pub backend: BackendPreference,
    pub max_batch_size: usize,
    pub num_threads: usize,
    pub max_concurrent_requests: usize,
    pub request_timeout_secs: u64,
    pub cors_enabled: bool,
    pub cors_origins: Vec<String>,
    pub ui_enabled: bool,
    pub ui_dir: PathBuf,
}

impl Default for ServeRuntimeConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            models_dir: default_models_dir(),
            backend: default_backend(),
            max_batch_size: default_max_batch_size(),
            num_threads: default_num_threads(),
            max_concurrent_requests: default_max_concurrent_requests(),
            request_timeout_secs: default_request_timeout_secs(),
            cors_enabled: default_cors_enabled(),
            cors_origins: default_cors_origins(),
            ui_enabled: default_ui_enabled(),
            ui_dir: default_ui_dir(),
        }
    }
}

impl ServeRuntimeConfig {
    pub fn from_sources(
        config_file: &ServeRuntimeConfigOverrides,
        env: &ServeRuntimeConfigOverrides,
        cli: &ServeRuntimeConfigOverrides,
    ) -> Self {
        Self::default()
            .apply_overrides(config_file)
            .apply_overrides(env)
            .apply_overrides(cli)
    }

    pub fn apply_overrides(mut self, overrides: &ServeRuntimeConfigOverrides) -> Self {
        if let Some(host) = overrides.host.as_ref() {
            self.host = host.clone();
        }
        if let Some(port) = overrides.port {
            self.port = port;
        }
        if let Some(models_dir) = overrides.models_dir.as_ref() {
            self.models_dir = models_dir.clone();
        }
        if let Some(backend) = overrides.backend {
            self.backend = backend;
        }
        if let Some(max_batch_size) = overrides.max_batch_size {
            self.max_batch_size = max_batch_size;
        }
        if let Some(num_threads) = overrides.num_threads {
            self.num_threads = num_threads;
        }
        if let Some(max_concurrent_requests) = overrides.max_concurrent_requests {
            self.max_concurrent_requests = max_concurrent_requests;
        }
        if let Some(request_timeout_secs) = overrides.request_timeout_secs {
            self.request_timeout_secs = request_timeout_secs;
        }
        if let Some(cors_enabled) = overrides.cors_enabled {
            self.cors_enabled = cors_enabled;
        }
        if let Some(cors_origins) = overrides.cors_origins.as_ref() {
            self.cors_origins = cors_origins.clone();
        }
        if let Some(ui_enabled) = overrides.ui_enabled {
            self.ui_enabled = ui_enabled;
        }
        if let Some(ui_dir) = overrides.ui_dir.as_ref() {
            self.ui_dir = ui_dir.clone();
        }

        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeRuntimeConfigOverrides {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub models_dir: Option<PathBuf>,
    pub backend: Option<BackendPreference>,
    pub max_batch_size: Option<usize>,
    pub num_threads: Option<usize>,
    pub max_concurrent_requests: Option<usize>,
    pub request_timeout_secs: Option<u64>,
    pub cors_enabled: Option<bool>,
    pub cors_origins: Option<Vec<String>>,
    pub ui_enabled: Option<bool>,
    pub ui_dir: Option<PathBuf>,
}

impl ServeRuntimeConfigOverrides {
    pub fn from_env() -> Self {
        Self {
            host: read_env_string(ENV_HOST, &[]),
            port: read_env_u16(ENV_PORT, &[]),
            models_dir: read_env_path(ENV_MODELS_DIR, &[]),
            backend: read_env_backend(ENV_BACKEND, &[]),
            max_batch_size: read_env_usize(ENV_MAX_BATCH_SIZE, &[]),
            num_threads: read_env_usize(ENV_NUM_THREADS, &[]),
            max_concurrent_requests: read_env_usize(ENV_MAX_CONCURRENT, LEGACY_ENV_MAX_CONCURRENT),
            request_timeout_secs: read_env_u64(ENV_TIMEOUT, LEGACY_ENV_TIMEOUT),
            cors_enabled: read_env_bool(ENV_CORS, &[]),
            cors_origins: read_env_csv(ENV_CORS_ORIGINS, &[]),
            ui_enabled: read_env_bool(ENV_NO_UI, &[]).map(|no_ui| !no_ui),
            ui_dir: read_env_path(ENV_UI_DIR, &[]),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

fn default_backend() -> BackendPreference {
    BackendPreference::Auto
}

fn default_max_batch_size() -> usize {
    8
}

fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8)
}

fn default_max_concurrent_requests() -> usize {
    100
}

fn default_request_timeout_secs() -> u64 {
    300
}

fn default_cors_enabled() -> bool {
    true
}

fn default_cors_origins() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_ui_enabled() -> bool {
    true
}

fn default_ui_dir() -> PathBuf {
    PathBuf::from("ui/dist")
}

fn first_non_empty_env(primary: &str, aliases: &[&str]) -> Option<String> {
    std::iter::once(primary)
        .chain(aliases.iter().copied())
        .find_map(|key| {
            std::env::var(key)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn read_env_string(primary: &str, aliases: &[&str]) -> Option<String> {
    first_non_empty_env(primary, aliases)
}

fn read_env_path(primary: &str, aliases: &[&str]) -> Option<PathBuf> {
    first_non_empty_env(primary, aliases).map(PathBuf::from)
}

fn read_env_backend(primary: &str, aliases: &[&str]) -> Option<BackendPreference> {
    first_non_empty_env(primary, aliases).and_then(|value| BackendPreference::parse(&value))
}

fn read_env_u16(primary: &str, aliases: &[&str]) -> Option<u16> {
    first_non_empty_env(primary, aliases).and_then(|value| value.parse::<u16>().ok())
}

fn read_env_u64(primary: &str, aliases: &[&str]) -> Option<u64> {
    first_non_empty_env(primary, aliases).and_then(|value| value.parse::<u64>().ok())
}

fn read_env_usize(primary: &str, aliases: &[&str]) -> Option<usize> {
    first_non_empty_env(primary, aliases)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn read_env_bool(primary: &str, aliases: &[&str]) -> Option<bool> {
    first_non_empty_env(primary, aliases).and_then(|value| {
        match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn read_env_csv(primary: &str, aliases: &[&str]) -> Option<Vec<String>> {
    first_non_empty_env(primary, aliases).map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_ENV_KEYS: &[&str] = &[
        ENV_HOST,
        ENV_PORT,
        ENV_MODELS_DIR,
        ENV_BACKEND,
        ENV_MAX_BATCH_SIZE,
        ENV_NUM_THREADS,
        ENV_MAX_CONCURRENT,
        ENV_TIMEOUT,
        ENV_CORS,
        ENV_CORS_ORIGINS,
        ENV_NO_UI,
        ENV_UI_DIR,
        LEGACY_ENV_MAX_CONCURRENT[0],
        LEGACY_ENV_TIMEOUT[0],
    ];

    fn clear_env() {
        for key in ALL_ENV_KEYS {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn env_overrides_accept_legacy_aliases() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var(LEGACY_ENV_MAX_CONCURRENT[0], "77");
        std::env::set_var(LEGACY_ENV_TIMEOUT[0], "555");

        let overrides = ServeRuntimeConfigOverrides::from_env();

        assert_eq!(overrides.max_concurrent_requests, Some(77));
        assert_eq!(overrides.request_timeout_secs, Some(555));
        clear_env();
    }

    #[test]
    fn canonical_env_keys_override_legacy_aliases() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var(LEGACY_ENV_MAX_CONCURRENT[0], "77");
        std::env::set_var(LEGACY_ENV_TIMEOUT[0], "555");
        std::env::set_var(ENV_MAX_CONCURRENT, "42");
        std::env::set_var(ENV_TIMEOUT, "123");

        let overrides = ServeRuntimeConfigOverrides::from_env();

        assert_eq!(overrides.max_concurrent_requests, Some(42));
        assert_eq!(overrides.request_timeout_secs, Some(123));
        clear_env();
    }

    #[test]
    fn resolve_uses_cli_then_env_then_config_then_defaults() {
        let config_file = ServeRuntimeConfigOverrides {
            host: Some("config-host".to_string()),
            port: Some(9001),
            max_batch_size: Some(6),
            num_threads: Some(3),
            max_concurrent_requests: Some(50),
            request_timeout_secs: Some(111),
            cors_enabled: Some(false),
            ui_enabled: Some(false),
            ..ServeRuntimeConfigOverrides::default()
        };
        let env = ServeRuntimeConfigOverrides {
            host: Some("env-host".to_string()),
            max_batch_size: Some(7),
            request_timeout_secs: Some(222),
            cors_enabled: Some(true),
            ..ServeRuntimeConfigOverrides::default()
        };
        let cli = ServeRuntimeConfigOverrides {
            host: Some("cli-host".to_string()),
            port: Some(9003),
            num_threads: Some(5),
            ui_enabled: Some(true),
            ..ServeRuntimeConfigOverrides::default()
        };

        let resolved = ServeRuntimeConfig::from_sources(&config_file, &env, &cli);

        assert_eq!(resolved.host, "cli-host");
        assert_eq!(resolved.port, 9003);
        assert_eq!(resolved.max_batch_size, 7);
        assert_eq!(resolved.num_threads, 5);
        assert_eq!(resolved.max_concurrent_requests, 50);
        assert_eq!(resolved.request_timeout_secs, 222);
        assert!(resolved.cors_enabled);
        assert!(resolved.ui_enabled);
    }

    #[test]
    fn env_bool_and_csv_parsing_supports_ui_and_cors_contract() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var(ENV_CORS, "true");
        std::env::set_var(
            ENV_CORS_ORIGINS,
            "http://localhost:3000, https://example.com",
        );
        std::env::set_var(ENV_NO_UI, "1");

        let overrides = ServeRuntimeConfigOverrides::from_env();

        assert_eq!(overrides.cors_enabled, Some(true));
        assert_eq!(
            overrides.cors_origins,
            Some(vec![
                "http://localhost:3000".to_string(),
                "https://example.com".to_string()
            ])
        );
        assert_eq!(overrides.ui_enabled, Some(false));
        clear_env();
    }
}

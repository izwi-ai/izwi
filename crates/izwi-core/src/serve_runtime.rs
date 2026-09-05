use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::backends::BackendPreference;
use crate::config::{
    BatchSizePreference, ContextLengthPreference, EngineConfig, PhysicalExecutionMode,
    PhysicalInFlightLimit,
};

pub const ENV_HOST: &str = "IZWI_HOST";
pub const ENV_PORT: &str = "IZWI_PORT";
pub const ENV_MODELS_DIR: &str = "IZWI_MODELS_DIR";
pub const ENV_MAX_LOADED_MODELS: &str = "IZWI_MAX_LOADED_MODELS";
pub const ENV_BACKEND: &str = "IZWI_BACKEND";
pub const ENV_MAX_BATCH_SIZE: &str = "IZWI_MAX_BATCH_SIZE";
pub const ENV_PHYSICAL_EXECUTION_MODE: &str = "IZWI_PHYSICAL_EXECUTION_MODE";
pub const ENV_MAX_PHYSICAL_IN_FLIGHT: &str = "IZWI_MAX_PHYSICAL_IN_FLIGHT";
pub const ENV_MAX_SCHEDULER_BATCH_SIZE: &str = "IZWI_MAX_SCHEDULER_BATCH_SIZE";
pub const ENV_ENABLE_PREFIX_CACHING: &str = "IZWI_ENABLE_PREFIX_CACHING";
pub const ENV_MANAGED_PREFIX_CACHE_SALT: &str = "IZWI_MANAGED_PREFIX_CACHE_SALT";
pub const ENV_MAX_PREFIX_CACHE_PAGES: &str = "IZWI_MAX_PREFIX_CACHE_PAGES";
pub const ENV_ENABLE_CHUNKED_PREFILL: &str = "IZWI_ENABLE_CHUNKED_PREFILL";
pub const ENV_CHUNKED_PREFILL_THRESHOLD: &str = "IZWI_CHUNKED_PREFILL_THRESHOLD";
pub const ENV_MAX_RETAINED_SEQUENCES: &str = "IZWI_MAX_RETAINED_SEQUENCES";
pub const ENV_MAX_STAGED_TRANSACTIONS: &str = "IZWI_MAX_STAGED_TRANSACTIONS";
pub const ENV_MAX_QUEUED_REQUESTS: &str = "IZWI_MAX_QUEUED_REQUESTS";
pub const ENV_MAX_SEQUENCE_LENGTH: &str = "IZWI_MAX_SEQUENCE_LENGTH";
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
    /// Performance policy, resolved once before model loading.
    #[serde(default)]
    pub performance: crate::performance::PerformanceConfig,
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
    pub max_loaded_models: usize,
    pub backend: BackendPreference,
    pub max_batch_size: BatchSizePreference,
    pub physical_execution_mode: PhysicalExecutionMode,
    pub max_physical_in_flight: PhysicalInFlightLimit,
    pub max_scheduler_batch_size: usize,
    pub enable_prefix_caching: bool,
    pub managed_prefix_cache_salt: Option<String>,
    pub max_prefix_cache_pages: usize,
    pub enable_chunked_prefill: bool,
    pub chunked_prefill_threshold: usize,
    pub max_retained_sequences: usize,
    pub max_staged_transactions: usize,
    pub max_queued_requests: usize,
    pub max_sequence_length: ContextLengthPreference,
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
            performance: Default::default(),
            host: default_host(),
            port: default_port(),
            models_dir: default_models_dir(),
            max_loaded_models: 1,
            backend: default_backend(),
            max_batch_size: default_max_batch_size(),
            physical_execution_mode: PhysicalExecutionMode::Serial,
            max_physical_in_flight: PhysicalInFlightLimit::default(),
            max_scheduler_batch_size: default_max_scheduler_batch_size(),
            enable_prefix_caching: default_enable_prefix_caching(),
            managed_prefix_cache_salt: default_managed_prefix_cache_salt(),
            max_prefix_cache_pages: default_max_prefix_cache_pages(),
            enable_chunked_prefill: default_enable_chunked_prefill(),
            chunked_prefill_threshold: default_chunked_prefill_threshold(),
            max_retained_sequences: default_max_retained_sequences(),
            max_staged_transactions: default_max_staged_transactions(),
            max_queued_requests: default_max_queued_requests(),
            max_sequence_length: ContextLengthPreference::Auto,
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
        let mut config = Self::default()
            .apply_overrides(config_file)
            .apply_overrides(env)
            .apply_overrides(cli);
        config.performance.mark_environment_resolved();
        config
    }

    pub fn apply_overrides(mut self, overrides: &ServeRuntimeConfigOverrides) -> Self {
        self.performance.apply_overrides(&overrides.performance);
        if let Some(host) = overrides.host.as_ref() {
            self.host = host.clone();
        }
        if let Some(port) = overrides.port {
            self.port = port;
        }
        if let Some(models_dir) = overrides.models_dir.as_ref() {
            self.models_dir = models_dir.clone();
        }
        if let Some(max_loaded_models) = overrides.max_loaded_models {
            self.max_loaded_models = max_loaded_models.max(1);
        }
        if let Some(backend) = overrides.backend {
            self.backend = backend;
        }
        if let Some(max_batch_size) = overrides.max_batch_size {
            self.max_batch_size = max_batch_size;
        }
        if let Some(physical_execution_mode) = overrides.physical_execution_mode {
            self.physical_execution_mode = physical_execution_mode;
        }
        if let Some(max_physical_in_flight) = overrides.max_physical_in_flight {
            self.max_physical_in_flight = max_physical_in_flight;
        }
        if let Some(max_scheduler_batch_size) = overrides.max_scheduler_batch_size {
            self.max_scheduler_batch_size = max_scheduler_batch_size;
        }
        if let Some(enable_prefix_caching) = overrides.enable_prefix_caching {
            self.enable_prefix_caching = enable_prefix_caching;
        }
        if let Some(managed_prefix_cache_salt) = overrides.managed_prefix_cache_salt.as_ref() {
            self.managed_prefix_cache_salt = Some(managed_prefix_cache_salt.clone());
        }
        if let Some(max_prefix_cache_pages) = overrides.max_prefix_cache_pages {
            self.max_prefix_cache_pages = max_prefix_cache_pages;
        }
        if let Some(enable_chunked_prefill) = overrides.enable_chunked_prefill {
            self.enable_chunked_prefill = enable_chunked_prefill;
        }
        if let Some(chunked_prefill_threshold) = overrides.chunked_prefill_threshold {
            self.chunked_prefill_threshold = chunked_prefill_threshold;
        }
        if let Some(max_retained_sequences) = overrides.max_retained_sequences {
            self.max_retained_sequences = max_retained_sequences;
        }
        if let Some(max_staged_transactions) = overrides.max_staged_transactions {
            self.max_staged_transactions = max_staged_transactions;
        }
        if let Some(max_queued_requests) = overrides.max_queued_requests {
            self.max_queued_requests = max_queued_requests;
        }
        if let Some(max_sequence_length) = overrides.max_sequence_length {
            self.max_sequence_length = max_sequence_length;
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

    pub fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            performance: self.performance.clone(),
            models_dir: self.models_dir.clone(),
            max_loaded_models: Some(self.max_loaded_models.max(1)),
            max_batch_size: self.max_batch_size,
            physical_execution_mode: self.physical_execution_mode,
            max_physical_in_flight: self.max_physical_in_flight,
            max_scheduler_batch_size: self.max_scheduler_batch_size.max(1),
            enable_prefix_caching: self.enable_prefix_caching,
            managed_prefix_cache_salt: self.managed_prefix_cache_salt.clone(),
            max_prefix_cache_pages: self.max_prefix_cache_pages,
            enable_chunked_prefill: self.enable_chunked_prefill,
            chunked_prefill_threshold: self.chunked_prefill_threshold.max(1),
            max_retained_sequences: self.max_retained_sequences.max(1),
            max_staged_transactions: self.max_staged_transactions.max(1),
            max_queued_requests: self.max_queued_requests.max(1),
            max_sequence_length: self.max_sequence_length,
            backend: self.backend,
            num_threads: self.num_threads.max(1),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeRuntimeConfigOverrides {
    #[serde(
        default,
        skip_serializing_if = "crate::performance::PerformanceConfigOverrides::is_empty"
    )]
    pub performance: crate::performance::PerformanceConfigOverrides,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub models_dir: Option<PathBuf>,
    pub max_loaded_models: Option<usize>,
    pub backend: Option<BackendPreference>,
    pub max_batch_size: Option<BatchSizePreference>,
    pub physical_execution_mode: Option<PhysicalExecutionMode>,
    pub max_physical_in_flight: Option<PhysicalInFlightLimit>,
    pub max_scheduler_batch_size: Option<usize>,
    pub enable_prefix_caching: Option<bool>,
    pub managed_prefix_cache_salt: Option<String>,
    pub max_prefix_cache_pages: Option<usize>,
    pub enable_chunked_prefill: Option<bool>,
    pub chunked_prefill_threshold: Option<usize>,
    pub max_retained_sequences: Option<usize>,
    pub max_staged_transactions: Option<usize>,
    pub max_queued_requests: Option<usize>,
    pub max_sequence_length: Option<ContextLengthPreference>,
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
            performance: crate::performance::PerformanceConfigOverrides::from_env(),
            host: read_env_string(ENV_HOST, &[]),
            port: read_env_u16(ENV_PORT, &[]),
            models_dir: read_env_path(ENV_MODELS_DIR, &[]),
            max_loaded_models: read_env_usize(ENV_MAX_LOADED_MODELS, &[]),
            backend: read_env_backend(ENV_BACKEND, &[]),
            max_batch_size: read_env_batch_size(ENV_MAX_BATCH_SIZE, &[]),
            physical_execution_mode: read_env_physical_execution_mode(
                ENV_PHYSICAL_EXECUTION_MODE,
                &[],
            ),
            max_physical_in_flight: read_env_physical_in_flight_limit(
                ENV_MAX_PHYSICAL_IN_FLIGHT,
                &[],
            ),
            max_scheduler_batch_size: read_env_usize(ENV_MAX_SCHEDULER_BATCH_SIZE, &[]),
            enable_prefix_caching: read_env_bool(ENV_ENABLE_PREFIX_CACHING, &[]),
            managed_prefix_cache_salt: read_env_string(ENV_MANAGED_PREFIX_CACHE_SALT, &[]),
            max_prefix_cache_pages: read_env_usize(ENV_MAX_PREFIX_CACHE_PAGES, &[]),
            enable_chunked_prefill: read_env_bool(ENV_ENABLE_CHUNKED_PREFILL, &[]),
            chunked_prefill_threshold: read_env_usize(ENV_CHUNKED_PREFILL_THRESHOLD, &[]),
            max_retained_sequences: read_env_usize(ENV_MAX_RETAINED_SEQUENCES, &[]),
            max_staged_transactions: read_env_usize(ENV_MAX_STAGED_TRANSACTIONS, &[]),
            max_queued_requests: read_env_usize(ENV_MAX_QUEUED_REQUESTS, &[]),
            max_sequence_length: read_env_context(ENV_MAX_SEQUENCE_LENGTH, &[]),
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

fn default_max_batch_size() -> BatchSizePreference {
    BatchSizePreference::Auto
}

fn default_max_scheduler_batch_size() -> usize {
    8
}

fn default_enable_prefix_caching() -> bool {
    false
}

fn default_managed_prefix_cache_salt() -> Option<String> {
    None
}

fn default_max_prefix_cache_pages() -> usize {
    128
}

fn default_enable_chunked_prefill() -> bool {
    false
}

fn default_chunked_prefill_threshold() -> usize {
    192
}

fn default_max_retained_sequences() -> usize {
    8
}

fn default_max_staged_transactions() -> usize {
    8
}

fn default_max_queued_requests() -> usize {
    128
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
    false
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

fn read_env_batch_size(primary: &str, aliases: &[&str]) -> Option<BatchSizePreference> {
    first_non_empty_env(primary, aliases).and_then(|value| value.parse().ok())
}

fn read_env_physical_execution_mode(
    primary: &str,
    aliases: &[&str],
) -> Option<PhysicalExecutionMode> {
    first_non_empty_env(primary, aliases).and_then(|value| value.parse().ok())
}

fn read_env_physical_in_flight_limit(
    primary: &str,
    aliases: &[&str],
) -> Option<PhysicalInFlightLimit> {
    first_non_empty_env(primary, aliases).and_then(|value| value.parse().ok())
}

fn read_env_context(primary: &str, aliases: &[&str]) -> Option<ContextLengthPreference> {
    first_non_empty_env(primary, aliases).and_then(|value| value.parse().ok())
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
        ENV_MAX_LOADED_MODELS,
        ENV_BACKEND,
        ENV_MAX_BATCH_SIZE,
        ENV_PHYSICAL_EXECUTION_MODE,
        ENV_MAX_PHYSICAL_IN_FLIGHT,
        ENV_MAX_SCHEDULER_BATCH_SIZE,
        ENV_ENABLE_PREFIX_CACHING,
        ENV_MANAGED_PREFIX_CACHE_SALT,
        ENV_MAX_PREFIX_CACHE_PAGES,
        ENV_ENABLE_CHUNKED_PREFILL,
        ENV_CHUNKED_PREFILL_THRESHOLD,
        ENV_MAX_RETAINED_SEQUENCES,
        ENV_MAX_STAGED_TRANSACTIONS,
        ENV_MAX_QUEUED_REQUESTS,
        ENV_MAX_SEQUENCE_LENGTH,
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
    fn server_profile_defaults_to_one_resident_model_and_allows_override() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();

        let defaults = ServeRuntimeConfig::default();
        assert_eq!(defaults.max_loaded_models, 1);
        assert_eq!(defaults.engine_config().max_loaded_models, Some(1));

        std::env::set_var(ENV_MAX_LOADED_MODELS, "3");
        let resolved = ServeRuntimeConfig::from_sources(
            &ServeRuntimeConfigOverrides::default(),
            &ServeRuntimeConfigOverrides::from_env(),
            &ServeRuntimeConfigOverrides::default(),
        );
        assert_eq!(resolved.max_loaded_models, 3);
        assert_eq!(resolved.engine_config().max_loaded_models, Some(3));
        clear_env();
    }

    #[test]
    fn context_preference_survives_env_precedence_and_engine_config() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var(ENV_MAX_SEQUENCE_LENGTH, "8192");
        let env = ServeRuntimeConfigOverrides::from_env();
        let cli = ServeRuntimeConfigOverrides {
            max_sequence_length: Some(ContextLengthPreference::Auto),
            ..ServeRuntimeConfigOverrides::default()
        };
        let resolved = ServeRuntimeConfig::from_sources(
            &ServeRuntimeConfigOverrides {
                max_sequence_length: Some(ContextLengthPreference::explicit(2048).unwrap()),
                ..ServeRuntimeConfigOverrides::default()
            },
            &env,
            &cli,
        );
        assert_eq!(resolved.max_sequence_length, ContextLengthPreference::Auto);
        assert_eq!(
            resolved.engine_config().max_sequence_length,
            ContextLengthPreference::Auto
        );
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
            max_batch_size: Some(BatchSizePreference::fixed(6).unwrap()),
            num_threads: Some(3),
            max_concurrent_requests: Some(50),
            request_timeout_secs: Some(111),
            cors_enabled: Some(false),
            ui_enabled: Some(false),
            ..ServeRuntimeConfigOverrides::default()
        };
        let env = ServeRuntimeConfigOverrides {
            host: Some("env-host".to_string()),
            max_batch_size: Some(BatchSizePreference::fixed(7).unwrap()),
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
        assert_eq!(resolved.max_batch_size.fixed_rows(), Some(7));
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

    #[test]
    fn env_exposes_prefix_cache_and_chunked_prefill_knobs() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var(ENV_ENABLE_PREFIX_CACHING, "true");
        std::env::set_var(ENV_MANAGED_PREFIX_CACHE_SALT, "tenant-a");
        std::env::set_var(ENV_MAX_PREFIX_CACHE_PAGES, "64");
        std::env::set_var(ENV_ENABLE_CHUNKED_PREFILL, "1");
        std::env::set_var(ENV_CHUNKED_PREFILL_THRESHOLD, "512");

        let resolved = ServeRuntimeConfig::from_sources(
            &ServeRuntimeConfigOverrides::default(),
            &ServeRuntimeConfigOverrides::from_env(),
            &ServeRuntimeConfigOverrides::default(),
        );
        let engine = resolved.engine_config();

        assert!(resolved.enable_prefix_caching);
        assert_eq!(
            resolved.managed_prefix_cache_salt.as_deref(),
            Some("tenant-a")
        );
        assert_eq!(resolved.max_prefix_cache_pages, 64);
        assert!(resolved.enable_chunked_prefill);
        assert_eq!(resolved.chunked_prefill_threshold, 512);
        assert!(engine.enable_prefix_caching);
        assert_eq!(
            engine.managed_prefix_cache_salt.as_deref(),
            Some("tenant-a")
        );
        assert_eq!(engine.max_prefix_cache_pages, 64);
        assert!(engine.enable_chunked_prefill);
        assert_eq!(engine.chunked_prefill_threshold, 512);
        clear_env();
    }

    #[test]
    fn engine_config_uses_runtime_contract_values() {
        let resolved = ServeRuntimeConfig {
            models_dir: PathBuf::from("/tmp/izwi-models"),
            backend: BackendPreference::Cpu,
            max_batch_size: BatchSizePreference::fixed(12).unwrap(),
            physical_execution_mode: PhysicalExecutionMode::Shadow,
            max_physical_in_flight: PhysicalInFlightLimit::new(3).unwrap(),
            max_scheduler_batch_size: 9,
            max_retained_sequences: 10,
            max_staged_transactions: 3,
            max_queued_requests: 77,
            num_threads: 6,
            ..ServeRuntimeConfig::default()
        };

        let engine = resolved.engine_config();

        assert_eq!(engine.models_dir, PathBuf::from("/tmp/izwi-models"));
        assert_eq!(engine.max_batch_size.fixed_rows(), Some(12));
        assert_eq!(
            engine.physical_execution_mode,
            PhysicalExecutionMode::Shadow
        );
        assert_eq!(engine.max_physical_in_flight.get(), 3);
        assert_eq!(engine.max_scheduler_batch_size, 9);
        assert_eq!(engine.max_retained_sequences, 10);
        assert_eq!(engine.max_staged_transactions, 3);
        assert_eq!(engine.max_queued_requests, 77);
        assert_eq!(engine.backend, BackendPreference::Cpu);
        assert_eq!(engine.num_threads, 6);
    }

    #[test]
    fn env_exposes_fail_closed_physical_execution_controls() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
        clear_env();
        std::env::set_var(ENV_PHYSICAL_EXECUTION_MODE, "shadow");
        std::env::set_var(ENV_MAX_PHYSICAL_IN_FLIGHT, "4");

        let resolved = ServeRuntimeConfig::from_sources(
            &ServeRuntimeConfigOverrides::default(),
            &ServeRuntimeConfigOverrides::from_env(),
            &ServeRuntimeConfigOverrides::default(),
        );
        assert_eq!(
            resolved.physical_execution_mode,
            PhysicalExecutionMode::Shadow
        );
        assert_eq!(resolved.max_physical_in_flight.get(), 4);
        let capacity = resolved
            .engine_config()
            .resolved_physical_execution_capacity();
        assert_eq!(capacity.candidate_dispatch_limit.get(), 4);
        assert_eq!(capacity.physical_launch_limit.get(), 1);
        clear_env();
    }
}

use anyhow::{anyhow, Result};
use izwi_core::backends::BackendPreference;
use izwi_core::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default, skip_serializing_if = "ServerConfig::is_empty")]
    pub server: ServerConfig,
    #[serde(default, skip_serializing_if = "ModelsConfig::is_empty")]
    pub models: ModelsConfig,
    #[serde(default, skip_serializing_if = "RuntimeConfig::is_empty")]
    pub runtime: RuntimeConfig,
    #[serde(default, skip_serializing_if = "UiConfig::is_empty")]
    pub ui: UiConfig,
    #[serde(default, skip_serializing_if = "DefaultsConfig::is_empty")]
    pub defaults: DefaultsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cors: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cors_origins: Option<Vec<String>>,
}

impl ServerConfig {
    fn is_empty(&self) -> bool {
        self.host.is_none()
            && self.port.is_none()
            && self.cors.is_none()
            && self.cors_origins.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
}

impl ModelsConfig {
    fn is_empty(&self) -> bool {
        self.dir.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendPreference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
}

impl RuntimeConfig {
    fn is_empty(&self) -> bool {
        self.backend.is_none()
            && self.max_batch_size.is_none()
            && self.threads.is_none()
            && self.max_concurrent.is_none()
            && self.timeout.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UiConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
}

impl UiConfig {
    fn is_empty(&self) -> bool {
        self.enabled.is_none() && self.dir.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DefaultsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

impl DefaultsConfig {
    fn is_empty(&self) -> bool {
        self.model.is_none() && self.speaker.is_none() && self.format.is_none()
    }
}

impl Config {
    pub fn load(path: Option<&PathBuf>) -> Result<Self> {
        let config_path = config_path(path);

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    pub fn save(&self, path: Option<&PathBuf>) -> Result<()> {
        let config_path = config_path(path);

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        Ok(())
    }

    pub fn default_template() -> Self {
        let defaults = ServeRuntimeConfig::default();
        Self {
            server: ServerConfig {
                host: Some(defaults.host),
                port: Some(defaults.port),
                cors: Some(defaults.cors_enabled),
                cors_origins: Some(defaults.cors_origins),
            },
            models: ModelsConfig {
                dir: Some(defaults.models_dir),
            },
            runtime: RuntimeConfig {
                backend: Some(defaults.backend),
                max_batch_size: Some(defaults.max_batch_size),
                threads: Some(defaults.num_threads),
                max_concurrent: Some(defaults.max_concurrent_requests),
                timeout: Some(defaults.request_timeout_secs),
            },
            ui: UiConfig {
                enabled: Some(defaults.ui_enabled),
                dir: Some(defaults.ui_dir),
            },
            defaults: DefaultsConfig::default(),
        }
    }

    pub fn serve_runtime_overrides(&self) -> ServeRuntimeConfigOverrides {
        let cors_origins = self.server.cors_origins.clone();
        let cors_enabled = match (self.server.cors, cors_origins.as_ref()) {
            (Some(enabled), _) => Some(enabled),
            (None, Some(origins)) if !origins.is_empty() => Some(true),
            _ => None,
        };

        ServeRuntimeConfigOverrides {
            host: self.server.host.clone(),
            port: self.server.port,
            models_dir: self.models.dir.clone(),
            backend: self.runtime.backend,
            max_batch_size: self.runtime.max_batch_size,
            num_threads: self.runtime.threads,
            max_concurrent_requests: self.runtime.max_concurrent,
            request_timeout_secs: self.runtime.timeout,
            cors_enabled,
            cors_origins,
            ui_enabled: self.ui.enabled,
            ui_dir: self.ui.dir.clone(),
        }
    }

    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "server.host" => self.server.host = Some(parse_string(value)?),
            "server.port" => self.server.port = Some(parse_u16(value)?),
            "server.cors" => self.server.cors = Some(parse_bool(value)?),
            "server.cors_origins" => self.server.cors_origins = Some(parse_string_list(value)?),
            "models.dir" => self.models.dir = Some(parse_path(value)?),
            "runtime.backend" => self.runtime.backend = Some(parse_backend(value)?),
            "runtime.max_batch_size" => self.runtime.max_batch_size = Some(parse_usize(value)?),
            "runtime.threads" => self.runtime.threads = Some(parse_usize(value)?),
            "runtime.max_concurrent" => self.runtime.max_concurrent = Some(parse_usize(value)?),
            "runtime.timeout" => self.runtime.timeout = Some(parse_u64(value)?),
            "ui.enabled" => self.ui.enabled = Some(parse_bool(value)?),
            "ui.dir" => self.ui.dir = Some(parse_path(value)?),
            "defaults.model" => self.defaults.model = Some(parse_string(value)?),
            "defaults.speaker" => self.defaults.speaker = Some(parse_string(value)?),
            "defaults.format" => self.defaults.format = Some(parse_string(value)?),
            _ => return Err(anyhow!("Unsupported config key '{}'", key)),
        }

        Ok(())
    }

    pub fn get_value(&self, key: &str) -> Option<toml::Value> {
        match key {
            "server.host" => self.server.host.clone().map(toml::Value::String),
            "server.port" => self
                .server
                .port
                .map(|value| toml::Value::Integer(value.into())),
            "server.cors" => self.server.cors.map(toml::Value::Boolean),
            "server.cors_origins" => self.server.cors_origins.as_ref().map(string_array_value),
            "models.dir" => self
                .models
                .dir
                .as_ref()
                .map(|value| toml::Value::String(value.display().to_string())),
            "runtime.backend" => self
                .runtime
                .backend
                .map(|value| toml::Value::String(value.as_str().to_string())),
            "runtime.max_batch_size" => self
                .runtime
                .max_batch_size
                .map(|value| toml::Value::Integer(value as i64)),
            "runtime.threads" => self
                .runtime
                .threads
                .map(|value| toml::Value::Integer(value as i64)),
            "runtime.max_concurrent" => self
                .runtime
                .max_concurrent
                .map(|value| toml::Value::Integer(value as i64)),
            "runtime.timeout" => self
                .runtime
                .timeout
                .map(|value| toml::Value::Integer(value as i64)),
            "ui.enabled" => self.ui.enabled.map(toml::Value::Boolean),
            "ui.dir" => self
                .ui
                .dir
                .as_ref()
                .map(|value| toml::Value::String(value.display().to_string())),
            "defaults.model" => self.defaults.model.clone().map(toml::Value::String),
            "defaults.speaker" => self.defaults.speaker.clone().map(toml::Value::String),
            "defaults.format" => self.defaults.format.clone().map(toml::Value::String),
            _ => None,
        }
    }

    pub fn default_value_for_key(key: &str) -> Option<toml::Value> {
        Self::default_template().get_value(key)
    }
}

fn config_path(path: Option<&PathBuf>) -> PathBuf {
    path.cloned().unwrap_or_else(|| {
        dirs::config_dir()
            .map(|p| p.join("izwi").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("config.toml"))
    })
}

fn parse_string(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(anyhow!("Value cannot be empty"))
    } else {
        Ok(trimmed.to_string())
    }
}

fn parse_path(value: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(parse_string(value)?))
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(anyhow!("Expected a boolean value")),
    }
}

fn parse_u16(value: &str) -> Result<u16> {
    value
        .trim()
        .parse::<u16>()
        .map_err(|_| anyhow!("Expected a valid 16-bit integer"))
}

fn parse_u64(value: &str) -> Result<u64> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| anyhow!("Expected a valid unsigned integer"))
}

fn parse_usize(value: &str) -> Result<usize> {
    value
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow!("Expected a positive integer"))
}

fn parse_backend(value: &str) -> Result<BackendPreference> {
    BackendPreference::parse(value)
        .ok_or_else(|| anyhow!("Expected one of: auto, cpu, metal, cuda"))
}

fn parse_string_list(value: &str) -> Result<Vec<String>> {
    let trimmed = value.trim();
    if trimmed.starts_with('[') {
        let parsed: toml::Value = format!("value = {trimmed}").parse()?;
        return parsed
            .get("value")
            .and_then(toml::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty())
            .ok_or_else(|| anyhow!("Expected a TOML string array"));
    }

    let values: Vec<String> = trimmed
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect();

    if values.is_empty() {
        Err(anyhow!("Expected at least one origin"))
    } else {
        Ok(values)
    }
}

fn string_array_value(values: &Vec<String>) -> toml::Value {
    toml::Value::Array(values.iter().cloned().map(toml::Value::String).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn serve_runtime_overrides_map_full_schema() {
        let config = Config {
            server: ServerConfig {
                host: Some("127.0.0.1".to_string()),
                port: Some(9090),
                cors: Some(true),
                cors_origins: Some(vec!["http://localhost:3000".to_string()]),
            },
            models: ModelsConfig {
                dir: Some(PathBuf::from("/tmp/models")),
            },
            runtime: RuntimeConfig {
                backend: Some(BackendPreference::Cpu),
                max_batch_size: Some(12),
                threads: Some(6),
                max_concurrent: Some(48),
                timeout: Some(720),
            },
            ui: UiConfig {
                enabled: Some(false),
                dir: Some(PathBuf::from("/tmp/ui")),
            },
            defaults: DefaultsConfig::default(),
        };

        let overrides = config.serve_runtime_overrides();

        assert_eq!(overrides.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(overrides.port, Some(9090));
        assert_eq!(overrides.models_dir, Some(PathBuf::from("/tmp/models")));
        assert_eq!(overrides.backend, Some(BackendPreference::Cpu));
        assert_eq!(overrides.max_batch_size, Some(12));
        assert_eq!(overrides.num_threads, Some(6));
        assert_eq!(overrides.max_concurrent_requests, Some(48));
        assert_eq!(overrides.request_timeout_secs, Some(720));
        assert_eq!(overrides.cors_enabled, Some(true));
        assert_eq!(
            overrides.cors_origins,
            Some(vec!["http://localhost:3000".to_string()])
        );
        assert_eq!(overrides.ui_enabled, Some(false));
        assert_eq!(overrides.ui_dir, Some(PathBuf::from("/tmp/ui")));
    }

    #[test]
    fn set_value_parses_typed_runtime_keys() {
        let mut config = Config::default();

        config
            .set_value("server.port", "9000")
            .expect("port should parse");
        config
            .set_value("runtime.backend", "cuda")
            .expect("backend should parse");
        config
            .set_value(
                "server.cors_origins",
                "http://localhost:3000,https://example.com",
            )
            .expect("origins should parse");
        config
            .set_value("ui.enabled", "false")
            .expect("bool should parse");

        assert_eq!(config.server.port, Some(9000));
        assert_eq!(config.runtime.backend, Some(BackendPreference::Cuda));
        assert_eq!(
            config.server.cors_origins,
            Some(vec![
                "http://localhost:3000".to_string(),
                "https://example.com".to_string()
            ])
        );
        assert_eq!(config.ui.enabled, Some(false));
    }

    #[test]
    fn save_and_load_round_trip_new_runtime_sections() {
        let dir = tempdir().expect("temp dir should be created");
        let path = dir.path().join("config.toml");
        let config = Config::default_template();

        config.save(Some(&path)).expect("config should save");
        let loaded = Config::load(Some(&path)).expect("config should load");

        assert_eq!(loaded.server.host, config.server.host);
        assert_eq!(loaded.runtime.max_batch_size, config.runtime.max_batch_size);
        assert_eq!(loaded.ui.enabled, config.ui.enabled);
    }
}

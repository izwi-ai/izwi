//! Per-engine performance policy. `Auto` selects supported implementations;
//! `Off` is an explicit fallback. Changing this policy requires model reload.
//!
//! Precedence is defaults < TOML < environment < CLI. Environment is captured
//! once, including errors, and never mutated by policy resolution.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, str::FromStr};

macro_rules! selector {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { #[default] Auto, $($variant),+ }
        impl FromStr for $name {
            type Err = Error;
            fn from_str(value: &str) -> Result<Self> {
                match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(Self::Auto),
                    $($text => Ok(Self::$variant),)+
                    _ => Err(Error::ConfigError(format!("invalid {} value: {value}", stringify!($name)))),
                }
            }
        }
    };
}
selector!(OptimizationMode { Off => "off" });
selector!(CudaProjectionBackend { Q8 => "q8", NativeFp8 => "native_fp8" });
selector!(LoadingIoStrategy { Mmap => "mmap", Sequential => "sequential" });
impl OptimizationMode {
    pub const fn enabled(self) -> bool {
        matches!(self, Self::Auto)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerformanceConfig {
    pub cuda: CudaPerformanceConfig,
    pub loading: LoadingPerformanceConfig,
    // A clone carries the same snapshot. Serialization intentionally starts a
    // new configuration lifetime and excludes process-local resolution state.
    #[serde(skip)]
    environment_resolved: bool,
    #[serde(skip)]
    environment_error: Option<String>,
}

// Equality describes the policy, independent of whether this process has
// already captured its environment. Round trips preserve the public contract.
impl PartialEq for PerformanceConfig {
    fn eq(&self, other: &Self) -> bool {
        self.cuda == other.cuda && self.loading == other.loading
    }
}
impl Eq for PerformanceConfig {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaPerformanceConfig {
    pub mode: OptimizationMode,
    pub projection_backend: CudaProjectionBackend,
    pub packed_projections: OptimizationMode,
    pub fused_decode: OptimizationMode,
    pub device_sampling: OptimizationMode,
    pub decode_graphs: OptimizationMode,
    pub mtp: OptimizationMode,
    pub mtp_quantum: OptimizationMode,
    #[serde(deserialize_with = "deserialize_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,
    pub mtp_adaptive: bool,
}
impl Default for CudaPerformanceConfig {
    fn default() -> Self {
        Self {
            mode: OptimizationMode::Auto,
            projection_backend: CudaProjectionBackend::Auto,
            packed_projections: OptimizationMode::Auto,
            fused_decode: OptimizationMode::Auto,
            device_sampling: OptimizationMode::Auto,
            decode_graphs: OptimizationMode::Auto,
            mtp: OptimizationMode::Auto,
            mtp_quantum: OptimizationMode::Auto,
            mtp_draft_tokens: 1,
            mtp_adaptive: true,
        }
    }
}
impl CudaPerformanceConfig {
    pub const fn enabled(&self) -> bool {
        self.mode.enabled()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadingPerformanceConfig {
    pub mode: OptimizationMode,
    pub derived_weight_cache: OptimizationMode,
    pub parallel_conversion: OptimizationMode,
    pub pinned_uploads: OptimizationMode,
    pub io_strategy: LoadingIoStrategy,
    pub workers: usize,
    #[serde(deserialize_with = "deserialize_max_staging_bytes")]
    pub max_staging_bytes: usize,
    pub cache_max_bytes: u64,
    pub cache_dir: Option<PathBuf>,
}
impl Default for LoadingPerformanceConfig {
    fn default() -> Self {
        Self {
            mode: OptimizationMode::Auto,
            derived_weight_cache: OptimizationMode::Auto,
            parallel_conversion: OptimizationMode::Auto,
            pinned_uploads: OptimizationMode::Auto,
            io_strategy: LoadingIoStrategy::Auto,
            workers: 0,
            max_staging_bytes: 256 * 1024 * 1024,
            cache_max_bytes: 64 * 1024 * 1024 * 1024,
            cache_dir: None,
        }
    }
}
impl LoadingPerformanceConfig {
    pub const fn enabled(&self) -> bool {
        self.mode.enabled()
    }
}

/// A partial layer: omitted siblings never reset earlier sources.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerformanceConfigOverrides {
    #[serde(skip_serializing_if = "CudaPerformanceConfigOverrides::is_empty")]
    pub cuda: CudaPerformanceConfigOverrides,
    #[serde(skip_serializing_if = "LoadingPerformanceConfigOverrides::is_empty")]
    pub loading: LoadingPerformanceConfigOverrides,
    #[serde(skip)]
    environment_error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaPerformanceConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection_backend: Option<CudaProjectionBackend>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packed_projections: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused_decode: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_sampling: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_graphs: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_quantum: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_mtp_draft_tokens")]
    pub mtp_draft_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_adaptive: Option<bool>,
}
impl CudaPerformanceConfigOverrides {
    pub fn is_empty(&self) -> bool {
        self.mode.is_none()
            && self.projection_backend.is_none()
            && self.packed_projections.is_none()
            && self.fused_decode.is_none()
            && self.device_sampling.is_none()
            && self.decode_graphs.is_none()
            && self.mtp.is_none()
            && self.mtp_quantum.is_none()
            && self.mtp_draft_tokens.is_none()
            && self.mtp_adaptive.is_none()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadingPerformanceConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_weight_cache: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_conversion: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pinned_uploads: Option<OptimizationMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub io_strategy: Option<LoadingIoStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_max_staging_bytes")]
    pub max_staging_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_max_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
}
impl LoadingPerformanceConfigOverrides {
    pub fn is_empty(&self) -> bool {
        self.mode.is_none()
            && self.derived_weight_cache.is_none()
            && self.parallel_conversion.is_none()
            && self.pinned_uploads.is_none()
            && self.io_strategy.is_none()
            && self.workers.is_none()
            && self.max_staging_bytes.is_none()
            && self.cache_max_bytes.is_none()
            && self.cache_dir.is_none()
    }
}

impl PerformanceConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(error) = &self.environment_error {
            return Err(Error::ConfigError(error.clone()));
        }
        validate_mtp_draft_tokens(self.cuda.mtp_draft_tokens)?;
        validate_max_staging_bytes(self.loading.max_staging_bytes)?;
        Ok(())
    }

    /// Snapshot aliases once; a resolved CLI layer cannot be overwritten here.
    pub fn resolve_env(self) -> Result<Self> {
        let config = self.snapshot_env();
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn snapshot_env(mut self) -> Self {
        if !self.environment_resolved {
            self.apply_overrides(&PerformanceConfigOverrides::from_env());
            self.environment_resolved = true;
        }
        self
    }

    pub(crate) fn mark_environment_resolved(&mut self) {
        self.environment_resolved = true;
    }

    /// Effective feature modes; retain requested backend and budget diagnostics.
    /// Apply only after all source layers have merged.
    pub fn normalized(mut self) -> Self {
        if !self.cuda.enabled() {
            self.cuda.packed_projections = OptimizationMode::Off;
            self.cuda.fused_decode = OptimizationMode::Off;
            self.cuda.device_sampling = OptimizationMode::Off;
            self.cuda.decode_graphs = OptimizationMode::Off;
            self.cuda.mtp = OptimizationMode::Off;
            self.cuda.mtp_quantum = OptimizationMode::Off;
        }
        if !self.loading.enabled() {
            self.loading.derived_weight_cache = OptimizationMode::Off;
            self.loading.parallel_conversion = OptimizationMode::Off;
            self.loading.pinned_uploads = OptimizationMode::Off;
        }
        self
    }

    pub fn apply_overrides(&mut self, overrides: &PerformanceConfigOverrides) {
        if let Some(value) = overrides.cuda.mode {
            self.cuda.mode = value;
        }
        if let Some(value) = overrides.cuda.projection_backend {
            self.cuda.projection_backend = value;
        }
        if let Some(value) = overrides.cuda.packed_projections {
            self.cuda.packed_projections = value;
        }
        if let Some(value) = overrides.cuda.fused_decode {
            self.cuda.fused_decode = value;
        }
        if let Some(value) = overrides.cuda.device_sampling {
            self.cuda.device_sampling = value;
        }
        if let Some(value) = overrides.cuda.decode_graphs {
            self.cuda.decode_graphs = value;
        }
        if let Some(value) = overrides.cuda.mtp {
            self.cuda.mtp = value;
        }
        if let Some(value) = overrides.cuda.mtp_quantum {
            self.cuda.mtp_quantum = value;
        }
        if let Some(value) = overrides.cuda.mtp_draft_tokens {
            self.cuda.mtp_draft_tokens = value;
        }
        if let Some(value) = overrides.cuda.mtp_adaptive {
            self.cuda.mtp_adaptive = value;
        }
        if let Some(value) = overrides.loading.mode {
            self.loading.mode = value;
        }
        if let Some(value) = overrides.loading.derived_weight_cache {
            self.loading.derived_weight_cache = value;
        }
        if let Some(value) = overrides.loading.parallel_conversion {
            self.loading.parallel_conversion = value;
        }
        if let Some(value) = overrides.loading.pinned_uploads {
            self.loading.pinned_uploads = value;
        }
        if let Some(value) = overrides.loading.io_strategy {
            self.loading.io_strategy = value;
        }
        if let Some(value) = overrides.loading.workers {
            self.loading.workers = value;
        }
        if let Some(value) = overrides.loading.max_staging_bytes {
            self.loading.max_staging_bytes = value;
        }
        if let Some(value) = overrides.loading.cache_max_bytes {
            self.loading.cache_max_bytes = value;
        }
        if let Some(value) = &overrides.loading.cache_dir {
            self.loading.cache_dir = Some(value.clone());
        }
        if let Some(error) = &overrides.environment_error {
            self.environment_error = Some(error.clone());
        }
    }
}

impl From<&PerformanceConfig> for PerformanceConfigOverrides {
    fn from(config: &PerformanceConfig) -> Self {
        Self {
            cuda: CudaPerformanceConfigOverrides {
                mode: Some(config.cuda.mode),
                projection_backend: Some(config.cuda.projection_backend),
                packed_projections: Some(config.cuda.packed_projections),
                fused_decode: Some(config.cuda.fused_decode),
                device_sampling: Some(config.cuda.device_sampling),
                decode_graphs: Some(config.cuda.decode_graphs),
                mtp: Some(config.cuda.mtp),
                mtp_quantum: Some(config.cuda.mtp_quantum),
                mtp_draft_tokens: Some(config.cuda.mtp_draft_tokens),
                mtp_adaptive: Some(config.cuda.mtp_adaptive),
            },
            loading: LoadingPerformanceConfigOverrides {
                mode: Some(config.loading.mode),
                derived_weight_cache: Some(config.loading.derived_weight_cache),
                parallel_conversion: Some(config.loading.parallel_conversion),
                pinned_uploads: Some(config.loading.pinned_uploads),
                io_strategy: Some(config.loading.io_strategy),
                workers: Some(config.loading.workers),
                max_staging_bytes: Some(config.loading.max_staging_bytes),
                cache_max_bytes: Some(config.loading.cache_max_bytes),
                cache_dir: config.loading.cache_dir.clone(),
            },
            environment_error: None,
        }
    }
}

impl PerformanceConfigOverrides {
    pub fn is_empty(&self) -> bool {
        self.cuda.is_empty() && self.loading.is_empty() && self.environment_error.is_none()
    }

    /// Compatibility wrapper retains invalid environment values as a deferred
    /// startup error because the server's existing `from_env` API is infallible.
    pub fn from_env() -> Self {
        match Self::try_from_env() {
            Ok(value) => value,
            Err(error) => Self {
                environment_error: Some(match error {
                    Error::ConfigError(message) => message,
                    other => other.to_string(),
                }),
                ..Self::default()
            },
        }
    }

    pub fn try_from_env() -> Result<Self> {
        let mut invalid_encoding = Vec::new();
        let result = Self::from_lookup(|key| match std::env::var(key) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                invalid_encoding.push(key.to_owned());
                None
            }
        });
        if !invalid_encoding.is_empty() {
            return Err(Error::ConfigError(format!(
                "performance environment values must be Unicode: {}",
                invalid_encoding.join(", ")
            )));
        }
        result
    }

    /// Injectable lookup also lets tests prove precedence without global env writes.
    pub fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        let mut overrides = Self::default();
        for binding in ENVIRONMENT_BINDINGS {
            let contextual_error = |error| {
                Error::ConfigError(format!("{} ({}): {error}", binding.canonical, binding.key))
            };
            let canonical = lookup(binding.canonical);
            let canonical = canonical
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty());
            let value = if let Some(value) = canonical {
                Some(value.to_owned())
            } else {
                let aliases: Vec<String> = binding
                    .aliases
                    .iter()
                    .filter_map(|key| lookup(key))
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty())
                    .collect();
                // Several old fused-kernel switches share one mode now. An
                // explicit legacy opt-out must not accidentally activate it.
                if binding.mode {
                    let parsed = aliases
                        .iter()
                        .map(|value| parse_env_mode(value))
                        .collect::<Result<Vec<_>>>()
                        .map_err(contextual_error)?;
                    if parsed.contains(&OptimizationMode::Off) {
                        Some("off".into())
                    } else if !parsed.is_empty() {
                        Some("auto".into())
                    } else {
                        None
                    }
                } else {
                    aliases.into_iter().next()
                }
            };
            if let Some(value) = value {
                let value = if binding.mode {
                    match parse_env_mode(&value).map_err(contextual_error)? {
                        OptimizationMode::Auto => "auto",
                        OptimizationMode::Off => "off",
                    }
                    .to_owned()
                } else {
                    value
                };
                overrides.set_value(binding.key, &value).map_err(|error| {
                    Error::ConfigError(format!("{} ({}): {error}", binding.canonical, binding.key))
                })?;
            }
        }
        Ok(overrides)
    }

    /// `key` is relative to `runtime.performance` (for example `cuda.mtp`).
    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        let value = value.trim();
        match key {
            "cuda.mode" => {
                self.cuda.mode = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.projection_backend" => {
                self.cuda.projection_backend = Some(
                    value
                        .parse::<CudaProjectionBackend>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.packed_projections" => {
                self.cuda.packed_projections = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.fused_decode" => {
                self.cuda.fused_decode = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.device_sampling" => {
                self.cuda.device_sampling = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.decode_graphs" => {
                self.cuda.decode_graphs = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.mtp" => {
                self.cuda.mtp = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.mtp_quantum" => {
                self.cuda.mtp_quantum = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "cuda.mtp_draft_tokens" => {
                self.cuda.mtp_draft_tokens =
                    Some(validate_mtp_draft_tokens(value.parse::<usize>().map_err(
                        |error| Error::ConfigError(format!("{key}: {error}")),
                    )?)?)
            }
            "cuda.mtp_adaptive" => self.cuda.mtp_adaptive = Some(parse_bool(value)?),
            "loading.mode" => {
                self.loading.mode = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.derived_weight_cache" => {
                self.loading.derived_weight_cache = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.parallel_conversion" => {
                self.loading.parallel_conversion = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.pinned_uploads" => {
                self.loading.pinned_uploads = Some(
                    value
                        .parse::<OptimizationMode>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.io_strategy" => {
                self.loading.io_strategy = Some(
                    value
                        .parse::<LoadingIoStrategy>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.workers" => {
                self.loading.workers = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.max_staging_bytes" => {
                self.loading.max_staging_bytes = Some(validate_max_staging_bytes(
                    value
                        .parse::<usize>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )?)
            }
            "loading.cache_max_bytes" => {
                self.loading.cache_max_bytes = Some(
                    value
                        .parse::<u64>()
                        .map_err(|error| Error::ConfigError(format!("{key}: {error}")))?,
                )
            }
            "loading.cache_dir" => self.loading.cache_dir = Some(PathBuf::from(value)),
            _ => {
                return Err(Error::ConfigError(format!(
                    "unsupported performance key: {key}"
                )))
            }
        }
        Ok(())
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(Error::ConfigError(format!("expected boolean, got {value}"))),
    }
}
fn parse_env_mode(value: &str) -> Result<OptimizationMode> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(OptimizationMode::Auto);
    }
    parse_bool(value).map(|enabled| {
        if enabled {
            OptimizationMode::Auto
        } else {
            OptimizationMode::Off
        }
    })
}

fn validate_mtp_draft_tokens(value: usize) -> Result<usize> {
    if (1..=3).contains(&value) {
        Ok(value)
    } else {
        Err(Error::ConfigError(
            "cuda.mtp_draft_tokens must be in 1..=3".into(),
        ))
    }
}
fn deserialize_mtp_draft_tokens<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<usize, D::Error> {
    validate_mtp_draft_tokens(usize::deserialize(d)?).map_err(serde::de::Error::custom)
}
fn deserialize_optional_mtp_draft_tokens<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<usize>, D::Error> {
    Option::<usize>::deserialize(d)?
        .map(validate_mtp_draft_tokens)
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn validate_max_staging_bytes(value: usize) -> Result<usize> {
    if value > 0 {
        Ok(value)
    } else {
        Err(Error::ConfigError(
            "loading.max_staging_bytes must be positive".into(),
        ))
    }
}
fn deserialize_max_staging_bytes<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<usize, D::Error> {
    validate_max_staging_bytes(usize::deserialize(d)?).map_err(serde::de::Error::custom)
}
fn deserialize_optional_max_staging_bytes<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<usize>, D::Error> {
    Option::<usize>::deserialize(d)?
        .map(validate_max_staging_bytes)
        .transpose()
        .map_err(serde::de::Error::custom)
}

/// Canonical environment names win over deprecated model-specific aliases.
/// Boolean legacy values (including explicit `0`) remain accepted for modes.
pub struct PerformanceEnvironmentBinding {
    pub key: &'static str,
    pub canonical: &'static str,
    pub aliases: &'static [&'static str],
    pub mode: bool,
}
pub const ENVIRONMENT_BINDINGS: &[PerformanceEnvironmentBinding] = &[
    PerformanceEnvironmentBinding {
        key: "cuda.mode",
        canonical: "IZWI_CUDA_MODE",
        aliases: &["IZWI_PERFORMANCE_CUDA_MODE", "IZWI_QWEN38_MODE"],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.projection_backend",
        canonical: "IZWI_CUDA_PROJECTION_BACKEND",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_PROJECTION_BACKEND",
            "IZWI_QWEN38_PROJECTION_BACKEND",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.packed_projections",
        canonical: "IZWI_CUDA_PACKED_PROJECTIONS",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_PACKED_PROJECTIONS",
            "IZWI_QWEN38_PACKED_PROJECTIONS",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.fused_decode",
        canonical: "IZWI_CUDA_FUSED_DECODE",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_FUSED_DECODE",
            "IZWI_QWEN38_FUSED_DECODE",
            "IZWI_QWEN38_CAUSAL_CONV_DECODE",
            "IZWI_QWEN38_DECODE_EPILOGUES",
            "IZWI_QWEN38_DELTANET_DECODE",
            "IZWI_QWEN38_TILED_RECURRENCE",
            "IZWI_QWEN38_ROPE_KERNEL",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.device_sampling",
        canonical: "IZWI_CUDA_DEVICE_SAMPLING",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_DEVICE_SAMPLING",
            "IZWI_QWEN38_DEVICE_SAMPLING",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.decode_graphs",
        canonical: "IZWI_CUDA_DECODE_GRAPHS",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_DECODE_GRAPHS",
            "IZWI_QWEN38_DECODE_GRAPHS",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.mtp",
        canonical: "IZWI_CUDA_MTP",
        aliases: &["IZWI_PERFORMANCE_CUDA_MTP", "IZWI_QWEN38_MTP"],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.mtp_quantum",
        canonical: "IZWI_CUDA_MTP_QUANTUM",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_MTP_QUANTUM",
            "IZWI_QWEN38_MTP_QUANTUM",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.mtp_draft_tokens",
        canonical: "IZWI_CUDA_MTP_DRAFT_TOKENS",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_MTP_DRAFT_TOKENS",
            "IZWI_QWEN38_MTP_DRAFT_TOKENS",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "cuda.mtp_adaptive",
        canonical: "IZWI_CUDA_MTP_ADAPTIVE",
        aliases: &[
            "IZWI_PERFORMANCE_CUDA_MTP_ADAPTIVE",
            "IZWI_QWEN38_MTP_ADAPTIVE",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "loading.mode",
        canonical: "IZWI_LOADING_MODE",
        aliases: &["IZWI_PERFORMANCE_LOADING_MODE", "IZWI_QWEN38_LOADING_MODE"],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "loading.derived_weight_cache",
        canonical: "IZWI_LOADING_DERIVED_WEIGHT_CACHE",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_DERIVED_WEIGHT_CACHE",
            "IZWI_QWEN38_LOADING_DERIVED_WEIGHT_CACHE",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "loading.parallel_conversion",
        canonical: "IZWI_LOADING_PARALLEL_CONVERSION",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_PARALLEL_CONVERSION",
            "IZWI_QWEN38_LOADING_PARALLEL_CONVERSION",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "loading.pinned_uploads",
        canonical: "IZWI_LOADING_PINNED_UPLOADS",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_PINNED_UPLOADS",
            "IZWI_QWEN38_LOADING_PINNED_UPLOADS",
        ],
        mode: true,
    },
    PerformanceEnvironmentBinding {
        key: "loading.io_strategy",
        canonical: "IZWI_LOADING_IO_STRATEGY",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_IO_STRATEGY",
            "IZWI_QWEN38_LOADING_IO_STRATEGY",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "loading.workers",
        canonical: "IZWI_LOADING_WORKERS",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_WORKERS",
            "IZWI_QWEN38_LOADING_WORKERS",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "loading.max_staging_bytes",
        canonical: "IZWI_LOADING_MAX_STAGING_BYTES",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_MAX_STAGING_BYTES",
            "IZWI_QWEN38_LOADING_MAX_STAGING_BYTES",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "loading.cache_max_bytes",
        canonical: "IZWI_LOADING_CACHE_MAX_BYTES",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_CACHE_MAX_BYTES",
            "IZWI_QWEN38_LOADING_CACHE_MAX_BYTES",
        ],
        mode: false,
    },
    PerformanceEnvironmentBinding {
        key: "loading.cache_dir",
        canonical: "IZWI_LOADING_CACHE_DIR",
        aliases: &[
            "IZWI_PERFORMANCE_LOADING_CACHE_DIR",
            "IZWI_QWEN38_LOADING_CACHE_DIR",
        ],
        mode: false,
    },
];

/// User configuration path shared by CLI and standalone server startup.
pub fn default_user_config_path() -> PathBuf {
    dirs::config_dir()
        .map(|directory| directory.join("izwi").join("config.toml"))
        .unwrap_or_else(|| PathBuf::from("config.toml"))
}

impl PerformanceConfigOverrides {
    /// Read just the typed performance section of the existing user TOML.
    /// Other CLI configuration sections keep their existing owners and schema.
    pub fn from_user_config(path: Option<&std::path::Path>) -> Result<Self> {
        let path = path
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(default_user_config_path);
        let source = match std::fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default())
            }
            Err(error) => return Err(Error::ConfigError(format!("{}: {error}", path.display()))),
        };
        let document: toml::Value = toml::from_str(&source)
            .map_err(|error| Error::ConfigError(format!("{}: {error}", path.display())))?;
        if document
            .get("runtime")
            .is_some_and(|runtime| !runtime.is_table())
        {
            return Err(Error::ConfigError(format!(
                "{} runtime must be a TOML table",
                path.display()
            )));
        }
        match document
            .get("runtime")
            .and_then(|runtime| runtime.get("performance"))
        {
            Some(value) => value.clone().try_into().map_err(|error| {
                Error::ConfigError(format!("{} runtime.performance: {error}", path.display()))
            }),
            None => Ok(Self::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineConfig, ServeRuntimeConfig, ServeRuntimeConfigOverrides};

    fn layer(entries: &[(&str, &str)]) -> PerformanceConfigOverrides {
        PerformanceConfigOverrides::from_lookup(|key| {
            entries
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.to_string())
        })
        .unwrap()
    }

    #[test]
    fn empty_config_is_default_on_with_bounded_loading() {
        let config: PerformanceConfig = toml::from_str("").unwrap();
        assert!(config.cuda.enabled());
        assert!(config.loading.enabled());
        assert_eq!(config, PerformanceConfig::default());
        for binding in ENVIRONMENT_BINDINGS.iter().filter(|binding| binding.mode) {
            let (group, field) = binding.key.split_once('.').unwrap();
            assert_eq!(serde_json::to_value(&config).unwrap()[group][field], "auto");
        }
        assert_eq!(config.cuda.mtp_draft_tokens, 1);
        assert!(config.cuda.mtp_adaptive);
        assert_eq!(config.loading.workers, 0);
        assert_eq!(config.loading.max_staging_bytes, 256 * 1024 * 1024);
        assert_eq!(config.loading.cache_max_bytes, 64 * 1024 * 1024 * 1024);
    }

    #[test]
    fn source_precedence_preserves_unspecified_siblings_and_false() {
        let file: ServeRuntimeConfigOverrides = toml::from_str(
            r#"
            [performance.cuda]
            packed_projections = "off"
            mtp_draft_tokens = 3
            [performance.loading]
            workers = 2
            cache_max_bytes = 123456
        "#,
        )
        .unwrap();
        let env = ServeRuntimeConfigOverrides {
            performance: layer(&[
                ("IZWI_CUDA_MTP_DRAFT_TOKENS", "2"),
                ("IZWI_CUDA_MTP_ADAPTIVE", "true"),
                ("IZWI_LOADING_WORKERS", "4"),
            ]),
            ..Default::default()
        };
        let mut cli = ServeRuntimeConfigOverrides::default();
        cli.performance.cuda.mtp_adaptive = Some(false);
        cli.performance.loading.workers = Some(0);
        let config = ServeRuntimeConfig::from_sources(&file, &env, &cli);
        let performance = config.engine_config().performance.resolve_env().unwrap();
        assert_eq!(performance.cuda.packed_projections, OptimizationMode::Off);
        assert_eq!(performance.cuda.mtp_draft_tokens, 2);
        assert!(!performance.cuda.mtp_adaptive);
        assert_eq!(performance.loading.workers, 0);
        assert_eq!(performance.loading.cache_max_bytes, 123456);
    }

    #[test]
    fn master_off_dominates_features_but_normalization_waits_for_final_layer() {
        let mut config = PerformanceConfig::default();
        config.apply_overrides(&layer(&[
            ("IZWI_CUDA_MODE", "off"),
            ("IZWI_LOADING_MODE", "off"),
        ]));
        config.apply_overrides(&layer(&[
            ("IZWI_CUDA_MTP", "auto"),
            ("IZWI_LOADING_DERIVED_WEIGHT_CACHE", "auto"),
        ]));
        let effective = config.clone().normalized();
        for binding in ENVIRONMENT_BINDINGS.iter().filter(|binding| binding.mode) {
            let (group, field) = binding.key.split_once('.').unwrap();
            assert_eq!(
                serde_json::to_value(&effective).unwrap()[group][field],
                "off"
            );
        }
        config.apply_overrides(&layer(&[("IZWI_CUDA_MODE", "auto")]));
        assert!(config.normalized().cuda.mtp.enabled());
    }

    #[test]
    fn canonical_alias_and_legacy_zero_precedence() {
        let legacy = layer(&[
            ("IZWI_QWEN38_PACKED_PROJECTIONS", "0"),
            ("IZWI_QWEN38_MTP", "false"),
        ]);
        assert_eq!(legacy.cuda.packed_projections, Some(OptimizationMode::Off));
        assert_eq!(legacy.cuda.mtp, Some(OptimizationMode::Off));
        let canonical = layer(&[
            ("IZWI_CUDA_PACKED_PROJECTIONS", "auto"),
            ("IZWI_QWEN38_PACKED_PROJECTIONS", "0"),
        ]);
        assert_eq!(
            canonical.cuda.packed_projections,
            Some(OptimizationMode::Auto)
        );
        let fused = layer(&[
            ("IZWI_QWEN38_CAUSAL_CONV_DECODE", "1"),
            ("IZWI_QWEN38_DECODE_EPILOGUES", "0"),
        ]);
        assert_eq!(fused.cuda.fused_decode, Some(OptimizationMode::Off));
        let canonical = layer(&[
            ("IZWI_CUDA_FUSED_DECODE", "auto"),
            ("IZWI_QWEN38_DECODE_EPILOGUES", "0"),
        ]);
        assert_eq!(canonical.cuda.fused_decode, Some(OptimizationMode::Auto));
    }

    #[test]
    fn each_environment_key_is_read_at_most_once() {
        let mut reads = std::collections::HashSet::new();
        PerformanceConfigOverrides::from_lookup(|key| {
            assert!(reads.insert(key.to_owned()), "duplicate lookup: {key}");
            None
        })
        .unwrap();
    }

    #[test]
    fn resolved_snapshot_survives_engine_and_registry_style_clones() {
        let mut config = PerformanceConfig::default();
        config.apply_overrides(&layer(&[("IZWI_CUDA_MTP", "off")]));
        config.mark_environment_resolved();
        let resolved = config.clone().resolve_env().unwrap();
        assert_eq!(resolved, config);
        assert_eq!(resolved.clone().resolve_env().unwrap(), config);
        let serialized = serde_json::to_value(&resolved).unwrap();
        assert!(serialized.get("environment_resolved").is_none());
        assert!(serialized.get("environment_error").is_none());
        assert_eq!(
            serde_json::from_value::<PerformanceConfig>(serialized).unwrap(),
            resolved
        );
    }

    #[test]
    fn malformed_values_are_errors_in_env_toml_and_programmatic_config() {
        for (key, value) in [
            ("IZWI_CUDA_MTP", "maybe"),
            ("IZWI_CUDA_MTP_DRAFT_TOKENS", "0"),
            ("IZWI_CUDA_MTP_DRAFT_TOKENS", "4"),
            ("IZWI_LOADING_WORKERS", "-1"),
            ("IZWI_LOADING_MAX_STAGING_BYTES", "0"),
            ("IZWI_LOADING_IO_STRATEGY", "random"),
            ("IZWI_CUDA_PROJECTION_BACKEND", "bf16"),
        ] {
            assert!(
                PerformanceConfigOverrides::from_lookup(
                    |name| (name == key).then(|| value.to_owned())
                )
                .is_err(),
                "{key}={value}"
            );
        }
        for source in [
            "[cuda]\nmtp_draft_tokens = 0",
            "[cuda]\nmtp_draft_tokens = 4",
            "[loading]\nmax_staging_bytes = 0",
            "[cuda]\nmtp = 'maybe'",
            "[cuda]\nunknown = 'auto'",
        ] {
            assert!(
                toml::from_str::<PerformanceConfig>(source).is_err(),
                "{source}"
            );
            assert!(
                toml::from_str::<PerformanceConfigOverrides>(source).is_err(),
                "{source}"
            );
        }
        let mut manual = PerformanceConfig::default();
        manual.cuda.mtp_draft_tokens = 4;
        assert!(manual.validate().is_err());
    }

    #[test]
    fn deferred_environment_errors_are_not_silently_defaulted() {
        let mut overrides = ServeRuntimeConfigOverrides::default();
        overrides.performance.environment_error = Some("invalid IZWI_CUDA_MTP_DRAFT_TOKENS".into());
        let config =
            ServeRuntimeConfig::from_sources(&Default::default(), &overrides, &Default::default());
        assert!(config
            .engine_config()
            .performance
            .resolve_env()
            .unwrap_err()
            .to_string()
            .contains("IZWI_CUDA_MTP_DRAFT_TOKENS"));
    }

    #[test]
    fn serialization_preserves_values_and_legacy_engine_configs_default_policy() {
        let source = r#"
            [cuda]
            mode = "off"
            projection_backend = "native_fp8"
            mtp_draft_tokens = 3
            mtp_adaptive = false
            [loading]
            workers = 0
            cache_max_bytes = 0
            cache_dir = "/tmp/cache directory"
            io_strategy = "sequential"
        "#;
        let config: PerformanceConfig = toml::from_str(source).unwrap();
        let copy: PerformanceConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(copy, config);
        for source in [
            include_str!("../tests/fixtures/engine-config-beta17.json"),
            include_str!("../tests/fixtures/engine-config-beta18.json"),
        ] {
            let legacy: EngineConfig = serde_json::from_str(source).unwrap();
            assert_eq!(legacy.performance, PerformanceConfig::default());
        }
        let core: crate::engine::EngineCoreConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(core.performance, PerformanceConfig::default());
        let mut legacy = serde_json::to_value(ServeRuntimeConfig::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("performance");
        let server: ServeRuntimeConfig = serde_json::from_value(legacy).unwrap();
        assert_eq!(server.performance, PerformanceConfig::default());
    }
}

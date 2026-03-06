use serde::{Deserialize, Serialize};

use super::capabilities::BackendCapabilities;
use super::device::DeviceProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl BackendPreference {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" => None,
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "metal" | "mps" => Some(Self::Metal),
            "cuda" | "gpu" => Some(Self::Cuda),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn requested_kind(self) -> Option<BackendKind> {
        match self {
            Self::Auto => None,
            Self::Cpu => Some(BackendKind::Cpu),
            Self::Metal => Some(BackendKind::Metal),
            Self::Cuda => Some(BackendKind::Cuda),
        }
    }
}

impl From<BackendKind> for BackendPreference {
    fn from(value: BackendKind) -> Self {
        match value {
            BackendKind::Cpu => Self::Cpu,
            BackendKind::Metal => Self::Metal,
            BackendKind::Cuda => Self::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelectionSource {
    Default,
    Config,
    Env,
    Cli,
}

impl BackendSelectionSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Config => "config",
            Self::Env => "environment",
            Self::Cli => "cli",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    CandleNative,
    CandleMetal,
    CandleCuda,
}

impl ExecutionBackend {
    pub fn kind(self) -> BackendKind {
        match self {
            Self::CandleNative => BackendKind::Cpu,
            Self::CandleMetal => BackendKind::Metal,
            Self::CandleCuda => BackendKind::Cuda,
        }
    }

    pub fn from_kind(kind: BackendKind) -> Self {
        match kind {
            BackendKind::Cpu => Self::CandleNative,
            BackendKind::Metal => Self::CandleMetal,
            BackendKind::Cuda => Self::CandleCuda,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendContext {
    pub preference: BackendPreference,
    pub source: BackendSelectionSource,
    pub capabilities: BackendCapabilities,
    pub device: DeviceProfile,
    pub backend_kind: BackendKind,
    pub execution_backend: ExecutionBackend,
    pub reason: String,
}

impl BackendContext {
    pub fn new(
        preference: BackendPreference,
        source: BackendSelectionSource,
        capabilities: BackendCapabilities,
        device: DeviceProfile,
        reason: impl Into<String>,
    ) -> Self {
        let backend_kind = BackendKind::from(device.kind);
        let execution_backend = ExecutionBackend::from_kind(backend_kind);

        Self {
            preference,
            source,
            capabilities,
            device,
            backend_kind,
            execution_backend,
            reason: reason.into(),
        }
    }

    pub fn matches_preference(&self) -> bool {
        self.preference
            .requested_kind()
            .map(|requested| requested == self.backend_kind)
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_preference_variants() {
        assert_eq!(
            BackendPreference::parse("auto"),
            Some(BackendPreference::Auto)
        );
        assert_eq!(
            BackendPreference::parse("cpu"),
            Some(BackendPreference::Cpu)
        );
        assert_eq!(
            BackendPreference::parse("metal"),
            Some(BackendPreference::Metal)
        );
        assert_eq!(
            BackendPreference::parse("mps"),
            Some(BackendPreference::Metal)
        );
        assert_eq!(
            BackendPreference::parse("cuda"),
            Some(BackendPreference::Cuda)
        );
        assert_eq!(
            BackendPreference::parse("gpu"),
            Some(BackendPreference::Cuda)
        );
    }

    #[test]
    fn parse_backend_preference_rejects_unknown_values() {
        assert_eq!(BackendPreference::parse(""), None);
        assert_eq!(BackendPreference::parse("unknown"), None);
    }

    #[test]
    fn execution_backend_round_trips_via_kind() {
        for backend in [
            ExecutionBackend::CandleNative,
            ExecutionBackend::CandleMetal,
            ExecutionBackend::CandleCuda,
        ] {
            assert_eq!(ExecutionBackend::from_kind(backend.kind()), backend);
        }
    }

    #[test]
    fn backend_kind_maps_to_preference() {
        assert_eq!(
            BackendPreference::from(BackendKind::Cpu),
            BackendPreference::Cpu
        );
        assert_eq!(
            BackendPreference::from(BackendKind::Metal),
            BackendPreference::Metal
        );
        assert_eq!(
            BackendPreference::from(BackendKind::Cuda),
            BackendPreference::Cuda
        );
    }
}

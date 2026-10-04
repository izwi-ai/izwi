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

/// An execution device assigned to a production worker by its supervisor.
///
/// Unlike [`BackendPreference`], an assignment is strict: selecting any other
/// backend or physical accelerator is an error rather than a fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeDeviceAssignment {
    Cpu,
    Metal {
        process_local_device_index: usize,
        expected_device_id: String,
    },
    Cuda {
        process_local_device_index: usize,
        expected_device_uuid: String,
    },
}

impl RuntimeDeviceAssignment {
    pub const fn backend_kind(&self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::Cpu,
            Self::Metal { .. } => BackendKind::Metal,
            Self::Cuda { .. } => BackendKind::Cuda,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Cpu => Ok(()),
            Self::Metal {
                expected_device_id, ..
            } if expected_device_id.trim().is_empty() => {
                Err("Metal assignment requires a non-empty stable device ID")
            }
            Self::Cuda {
                expected_device_uuid,
                ..
            } if expected_device_uuid.trim().is_empty() => {
                Err("CUDA assignment requires a non-empty stable device UUID")
            }
            Self::Metal { .. } | Self::Cuda { .. } => Ok(()),
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

    #[test]
    fn runtime_assignments_are_strict_and_require_accelerator_identity() {
        assert_eq!(
            RuntimeDeviceAssignment::Cpu.backend_kind(),
            BackendKind::Cpu
        );
        assert!(RuntimeDeviceAssignment::Metal {
            process_local_device_index: 0,
            expected_device_id: " ".into(),
        }
        .validate()
        .is_err());
        assert!(RuntimeDeviceAssignment::Cuda {
            process_local_device_index: 0,
            expected_device_uuid: "GPU-0123".into(),
        }
        .validate()
        .is_ok());
    }
}

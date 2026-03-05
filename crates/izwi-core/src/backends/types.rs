use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
    Mlx,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Mlx => "mlx",
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelectionSource {
    Default,
    Config,
    Env,
    Cli,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    CandleNative,
    CandleMetal,
    CandleCuda,
    MlxNative,
}

impl ExecutionBackend {
    pub fn kind(self) -> BackendKind {
        match self {
            Self::CandleNative => BackendKind::Cpu,
            Self::CandleMetal => BackendKind::Metal,
            Self::CandleCuda => BackendKind::Cuda,
            Self::MlxNative => BackendKind::Mlx,
        }
    }

    pub fn from_kind(kind: BackendKind) -> Self {
        match kind {
            BackendKind::Cpu => Self::CandleNative,
            BackendKind::Metal => Self::CandleMetal,
            BackendKind::Cuda => Self::CandleCuda,
            BackendKind::Mlx => Self::MlxNative,
        }
    }
}

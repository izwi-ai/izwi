use super::types::BackendKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub cpu_compiled: bool,
    pub metal_compiled: bool,
    pub cuda_compiled: bool,
}

impl BackendCapabilities {
    pub fn detect() -> Self {
        Self {
            cpu_compiled: true,
            metal_compiled: cfg!(feature = "metal"),
            cuda_compiled: cfg!(feature = "cuda"),
        }
    }

    pub fn is_compiled_for(self, kind: BackendKind) -> bool {
        match kind {
            BackendKind::Cpu => self.cpu_compiled,
            BackendKind::Metal => self.metal_compiled,
            BackendKind::Cuda => self.cuda_compiled,
            // MLX support is not implemented yet.
            BackendKind::Mlx => false,
        }
    }
}

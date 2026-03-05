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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_is_always_compiled() {
        let caps = BackendCapabilities::detect();
        assert!(caps.is_compiled_for(BackendKind::Cpu));
    }

    #[test]
    fn metal_and_cuda_compile_flags_reflect_features() {
        let caps = BackendCapabilities::detect();
        assert_eq!(
            caps.is_compiled_for(BackendKind::Metal),
            cfg!(feature = "metal")
        );
        assert_eq!(
            caps.is_compiled_for(BackendKind::Cuda),
            cfg!(feature = "cuda")
        );
    }
}

//! Pure CUDA eligibility and runtime-resolution contracts.
//!
//! Eligibility is derived only from catalog policy. Runtime resolution consumes
//! explicitly observed facts and never upgrades catalog evidence by itself.

use serde::Serialize;

use crate::catalog::{
    CudaEvidenceLevel, CudaExecutionStatus, CudaQuantizationInfo, CudaQuantizationSupportLevel,
    CudaSupportInfo, CudaSupportLevel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaPlanBlocker {
    CatalogDisabled,
    CpuOnly,
    UnknownSupport,
    InconsistentEvidence,
    QuantizationUnsupported,
    CudaNotCompiled,
    CudaBackendNotSelected,
    CudaDeviceNotObserved,
}

impl CudaPlanBlocker {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CatalogDisabled => "catalog_disabled",
            Self::CpuOnly => "cpu_only",
            Self::UnknownSupport => "unknown_support",
            Self::InconsistentEvidence => "inconsistent_evidence",
            Self::QuantizationUnsupported => "quantization_unsupported",
            Self::CudaNotCompiled => "cuda_not_compiled",
            Self::CudaBackendNotSelected => "cuda_backend_not_selected",
            Self::CudaDeviceNotObserved => "cuda_device_not_observed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CudaPlanEligibility {
    pub eligible: bool,
    pub execution_status: CudaExecutionStatus,
    pub evidence: CudaEvidenceLevel,
    pub blocker: Option<CudaPlanBlocker>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaRuntimeObservation {
    pub cuda_compiled: bool,
    pub cuda_backend_selected: bool,
    pub cuda_device_observed: bool,
    pub device_name: Option<String>,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CudaExecutionPlan {
    pub eligibility: CudaPlanEligibility,
    pub executable: bool,
    pub runtime_blocker: Option<CudaPlanBlocker>,
    pub cuda_compiled: bool,
    pub cuda_backend_selected: bool,
    pub cuda_device_observed: bool,
    pub device_name: Option<String>,
    pub compute_capability: Option<(u32, u32)>,
}

pub fn evaluate_cuda_plan_eligibility(
    support: CudaSupportInfo,
    quantization: CudaQuantizationInfo,
) -> CudaPlanEligibility {
    let blocker = match support.level {
        CudaSupportLevel::Disabled => Some(CudaPlanBlocker::CatalogDisabled),
        CudaSupportLevel::CpuOnly => Some(CudaPlanBlocker::CpuOnly),
        CudaSupportLevel::Unknown => Some(CudaPlanBlocker::UnknownSupport),
        CudaSupportLevel::NativeCuda | CudaSupportLevel::CandleCudaGeneric
            if !support.evidence_is_sufficient() =>
        {
            Some(CudaPlanBlocker::InconsistentEvidence)
        }
        CudaSupportLevel::NativeCuda | CudaSupportLevel::CandleCudaGeneric => {
            match quantization.level {
                CudaQuantizationSupportLevel::CpuOnly
                | CudaQuantizationSupportLevel::Disabled
                | CudaQuantizationSupportLevel::Unknown => {
                    Some(CudaPlanBlocker::QuantizationUnsupported)
                }
                CudaQuantizationSupportLevel::Dense
                | CudaQuantizationSupportLevel::CandleQuantizedGeneric
                | CudaQuantizationSupportLevel::DenseDequantizedFallback => None,
            }
        }
    };

    CudaPlanEligibility {
        eligible: blocker.is_none(),
        execution_status: support.execution_status,
        evidence: support.evidence,
        blocker,
    }
}

pub fn resolve_cuda_execution_plan(
    eligibility: CudaPlanEligibility,
    runtime: CudaRuntimeObservation,
) -> CudaExecutionPlan {
    let runtime_blocker = if !eligibility.eligible {
        None
    } else if !runtime.cuda_compiled {
        Some(CudaPlanBlocker::CudaNotCompiled)
    } else if !runtime.cuda_backend_selected {
        Some(CudaPlanBlocker::CudaBackendNotSelected)
    } else if !runtime.cuda_device_observed {
        Some(CudaPlanBlocker::CudaDeviceNotObserved)
    } else {
        None
    };
    let (device_name, compute_capability) = if runtime.cuda_device_observed {
        (runtime.device_name, runtime.compute_capability)
    } else {
        (None, None)
    };

    CudaExecutionPlan {
        eligibility,
        executable: eligibility.eligible && runtime_blocker.is_none(),
        runtime_blocker,
        cuda_compiled: runtime.cuda_compiled,
        cuda_backend_selected: runtime.cuda_backend_selected,
        cuda_device_observed: runtime.cuda_device_observed,
        device_name,
        compute_capability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(level: CudaSupportLevel) -> CudaSupportInfo {
        CudaSupportInfo::new(level, "test support")
    }

    fn quantization(level: CudaQuantizationSupportLevel) -> CudaQuantizationInfo {
        CudaQuantizationInfo::new(level, "test quantization")
    }

    fn eligible() -> CudaPlanEligibility {
        evaluate_cuda_plan_eligibility(
            support(CudaSupportLevel::CandleCudaGeneric),
            quantization(CudaQuantizationSupportLevel::Dense),
        )
    }

    fn runtime(
        cuda_compiled: bool,
        cuda_backend_selected: bool,
        cuda_device_observed: bool,
    ) -> CudaRuntimeObservation {
        CudaRuntimeObservation {
            cuda_compiled,
            cuda_backend_selected,
            cuda_device_observed,
            device_name: cuda_device_observed.then(|| "test gpu".to_string()),
            compute_capability: cuda_device_observed.then_some((8, 0)),
        }
    }

    #[test]
    fn generic_dense_model_is_source_eligible_but_unverified() {
        let eligibility = eligible();
        assert!(eligibility.eligible);
        assert_eq!(eligibility.blocker, None);
        assert_eq!(
            eligibility.execution_status,
            CudaExecutionStatus::EligibleUnverified
        );
        assert_eq!(eligibility.evidence, CudaEvidenceLevel::SourceReviewed);
    }

    #[test]
    fn support_blockers_take_precedence_over_quantization() {
        for (level, expected) in [
            (CudaSupportLevel::Disabled, CudaPlanBlocker::CatalogDisabled),
            (CudaSupportLevel::CpuOnly, CudaPlanBlocker::CpuOnly),
            (CudaSupportLevel::Unknown, CudaPlanBlocker::UnknownSupport),
        ] {
            let eligibility = evaluate_cuda_plan_eligibility(
                support(level),
                quantization(CudaQuantizationSupportLevel::Unknown),
            );
            assert!(!eligibility.eligible);
            assert_eq!(eligibility.blocker, Some(expected));
        }
    }

    #[test]
    fn unsupported_quantization_blocks_an_eligible_model() {
        for level in [
            CudaQuantizationSupportLevel::CpuOnly,
            CudaQuantizationSupportLevel::Disabled,
            CudaQuantizationSupportLevel::Unknown,
        ] {
            let eligibility = evaluate_cuda_plan_eligibility(
                support(CudaSupportLevel::CandleCudaGeneric),
                quantization(level),
            );
            assert!(!eligibility.eligible);
            assert_eq!(
                eligibility.blocker,
                Some(CudaPlanBlocker::QuantizationUnsupported)
            );
        }
    }

    #[test]
    fn inconsistent_evidence_blocks_a_fabricated_optimized_claim() {
        let support = CudaSupportInfo {
            level: CudaSupportLevel::NativeCuda,
            execution_status: CudaExecutionStatus::CandleOptimized,
            evidence: CudaEvidenceLevel::SourceReviewed,
            reason: "fabricated optimized claim",
        };
        let eligibility = evaluate_cuda_plan_eligibility(
            support,
            quantization(CudaQuantizationSupportLevel::Dense),
        );

        assert!(!eligibility.eligible);
        assert_eq!(
            eligibility.blocker,
            Some(CudaPlanBlocker::InconsistentEvidence)
        );
    }

    #[test]
    fn runtime_blockers_have_deterministic_precedence() {
        for (observation, expected) in [
            (
                runtime(false, false, false),
                CudaPlanBlocker::CudaNotCompiled,
            ),
            (
                runtime(true, false, false),
                CudaPlanBlocker::CudaBackendNotSelected,
            ),
            (
                runtime(true, true, false),
                CudaPlanBlocker::CudaDeviceNotObserved,
            ),
        ] {
            let plan = resolve_cuda_execution_plan(eligible(), observation);
            assert!(!plan.executable);
            assert_eq!(plan.runtime_blocker, Some(expected));
        }
    }

    #[test]
    fn observed_runtime_does_not_upgrade_catalog_evidence() {
        let plan = resolve_cuda_execution_plan(eligible(), runtime(true, true, true));
        assert!(plan.executable);
        assert_eq!(plan.runtime_blocker, None);
        assert_eq!(
            plan.eligibility.execution_status,
            CudaExecutionStatus::EligibleUnverified
        );
        assert_eq!(plan.eligibility.evidence, CudaEvidenceLevel::SourceReviewed);
    }

    #[test]
    fn catalog_ineligibility_is_not_mislabeled_as_a_runtime_failure() {
        let eligibility = evaluate_cuda_plan_eligibility(
            support(CudaSupportLevel::CpuOnly),
            quantization(CudaQuantizationSupportLevel::CpuOnly),
        );
        let plan = resolve_cuda_execution_plan(eligibility, runtime(true, true, true));
        assert!(!plan.executable);
        assert_eq!(plan.runtime_blocker, None);
        assert_eq!(plan.eligibility.blocker, Some(CudaPlanBlocker::CpuOnly));
    }
}

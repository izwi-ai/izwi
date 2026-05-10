//! Stable runtime trace contracts.
//!
//! These names are intentionally backend-neutral so local, gateway, and worker
//! modes can emit comparable traces as execution moves behind the broker.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeTracePhase {
    BrokerAdmission,
    ModelLoad,
    SchedulerWait,
    Prefill,
    Decode,
    FirstChunk,
    PipelineStage,
    Cancellation,
    ModelUnload,
}

impl RuntimeTracePhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BrokerAdmission => "broker_admission",
            Self::ModelLoad => "model_load",
            Self::SchedulerWait => "scheduler_wait",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::FirstChunk => "first_chunk",
            Self::PipelineStage => "pipeline_stage",
            Self::Cancellation => "cancellation",
            Self::ModelUnload => "model_unload",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeTraceContract {
    pub phase: RuntimeTracePhase,
    pub span_name: &'static str,
    pub required_attributes: &'static [&'static str],
}

pub const TRACE_REQUEST_ID: &str = "izwi.request.id";
pub const TRACE_CORRELATION_ID: &str = "izwi.correlation.id";
pub const TRACE_CAPABILITY: &str = "izwi.capability";
pub const TRACE_MODEL_VARIANT: &str = "izwi.model.variant";
pub const TRACE_EXECUTION_TARGET: &str = "izwi.execution.target";
pub const TRACE_STREAMING_MODE: &str = "izwi.streaming.mode";
pub const TRACE_PIPELINE_KIND: &str = "izwi.pipeline.kind";
pub const TRACE_PIPELINE_STAGE: &str = "izwi.pipeline.stage";
pub const TRACE_ERROR_KIND: &str = "izwi.error.kind";

const REQUEST_CAPABILITY_ATTRIBUTES: &[&str] =
    &[TRACE_REQUEST_ID, TRACE_CORRELATION_ID, TRACE_CAPABILITY];
const MODEL_ATTRIBUTES: &[&str] = &[TRACE_MODEL_VARIANT];
const EXECUTION_ATTRIBUTES: &[&str] = &[
    TRACE_REQUEST_ID,
    TRACE_CAPABILITY,
    TRACE_MODEL_VARIANT,
    TRACE_EXECUTION_TARGET,
];
const STREAM_ATTRIBUTES: &[&str] = &[
    TRACE_REQUEST_ID,
    TRACE_CAPABILITY,
    TRACE_MODEL_VARIANT,
    TRACE_STREAMING_MODE,
];
const PIPELINE_ATTRIBUTES: &[&str] = &[
    TRACE_REQUEST_ID,
    TRACE_CAPABILITY,
    TRACE_PIPELINE_KIND,
    TRACE_PIPELINE_STAGE,
];
const ERROR_ATTRIBUTES: &[&str] = &[
    TRACE_REQUEST_ID,
    TRACE_CAPABILITY,
    TRACE_MODEL_VARIANT,
    TRACE_ERROR_KIND,
];

pub const RUNTIME_TRACE_CONTRACTS: &[RuntimeTraceContract] = &[
    RuntimeTraceContract {
        phase: RuntimeTracePhase::BrokerAdmission,
        span_name: "runtime.broker.admission",
        required_attributes: REQUEST_CAPABILITY_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::ModelLoad,
        span_name: "runtime.model.load",
        required_attributes: MODEL_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::SchedulerWait,
        span_name: "runtime.scheduler.wait",
        required_attributes: EXECUTION_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::Prefill,
        span_name: "runtime.engine.prefill",
        required_attributes: EXECUTION_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::Decode,
        span_name: "runtime.engine.decode",
        required_attributes: EXECUTION_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::FirstChunk,
        span_name: "runtime.stream.first_chunk",
        required_attributes: STREAM_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::PipelineStage,
        span_name: "runtime.pipeline.stage",
        required_attributes: PIPELINE_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::Cancellation,
        span_name: "runtime.request.cancellation",
        required_attributes: ERROR_ATTRIBUTES,
    },
    RuntimeTraceContract {
        phase: RuntimeTracePhase::ModelUnload,
        span_name: "runtime.model.unload",
        required_attributes: MODEL_ATTRIBUTES,
    },
];

pub fn runtime_trace_contracts() -> &'static [RuntimeTraceContract] {
    RUNTIME_TRACE_CONTRACTS
}

pub fn trace_contract_for_phase(
    phase: RuntimeTracePhase,
) -> Option<&'static RuntimeTraceContract> {
    runtime_trace_contracts()
        .iter()
        .find(|contract| contract.phase == phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn trace_contracts_cover_required_runtime_phases() {
        let phases = runtime_trace_contracts()
            .iter()
            .map(|contract| contract.phase)
            .collect::<BTreeSet<_>>();

        for phase in [
            RuntimeTracePhase::BrokerAdmission,
            RuntimeTracePhase::ModelLoad,
            RuntimeTracePhase::SchedulerWait,
            RuntimeTracePhase::Prefill,
            RuntimeTracePhase::Decode,
            RuntimeTracePhase::FirstChunk,
            RuntimeTracePhase::PipelineStage,
            RuntimeTracePhase::Cancellation,
            RuntimeTracePhase::ModelUnload,
        ] {
            assert!(
                phases.contains(&phase),
                "missing trace contract for {}",
                phase.as_str()
            );
        }
    }

    #[test]
    fn trace_contract_span_names_are_unique_and_namespaced() {
        let mut names = BTreeSet::new();

        for contract in runtime_trace_contracts() {
            assert!(
                contract.span_name.starts_with("runtime."),
                "trace span {} is not runtime namespaced",
                contract.span_name
            );
            assert!(
                names.insert(contract.span_name),
                "duplicate trace span {}",
                contract.span_name
            );
        }
    }

    #[test]
    fn request_scoped_trace_contracts_include_request_id() {
        for contract in runtime_trace_contracts() {
            if matches!(
                contract.phase,
                RuntimeTracePhase::ModelLoad | RuntimeTracePhase::ModelUnload
            ) {
                continue;
            }

            assert!(
                contract.required_attributes.contains(&TRACE_REQUEST_ID),
                "request-scoped trace {} must include request id",
                contract.span_name
            );
        }
    }
}

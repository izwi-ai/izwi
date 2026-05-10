//! Runtime telemetry ownership.

mod metrics;

pub(crate) use metrics::{push_engine_metric, RuntimeTelemetryCollector};
pub use metrics::{
    EngineRuntimeTelemetrySnapshot, InferenceBrokerRuntimeTelemetrySnapshot,
    PipelineRuntimeTelemetrySnapshot, RuntimeTelemetrySnapshot, VoiceRuntimeTelemetrySnapshot,
};

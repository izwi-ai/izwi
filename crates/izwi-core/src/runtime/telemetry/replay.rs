//! Sanitized replay metadata contracts.
//!
//! Replay records are intentionally payload-free. They capture enough metadata
//! to reproduce routing, scheduling, and telemetry timelines without retaining
//! user prompt text, chat messages, transcript text, or audio samples.

use crate::catalog::ModelVariant;
use crate::runtime::ConformanceCapability;
use crate::runtime::telemetry::RuntimeTracePhase;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayRedaction {
    SanitizedMetadataOnly,
}

impl ReplayRedaction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SanitizedMetadataOnly => "sanitized_metadata_only",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeReplayRecord {
    pub request_id: String,
    pub correlation_id: Option<String>,
    pub capability: ConformanceCapability,
    pub model_variant: Option<ModelVariant>,
    pub phase: RuntimeTracePhase,
    pub redaction: ReplayRedaction,
}

impl RuntimeReplayRecord {
    pub fn sanitized(
        request_id: impl Into<String>,
        capability: ConformanceCapability,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            correlation_id: None,
            capability,
            model_variant: None,
            phase: RuntimeTracePhase::BrokerAdmission,
            redaction: ReplayRedaction::SanitizedMetadataOnly,
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    pub fn with_model_variant(mut self, model_variant: ModelVariant) -> Self {
        self.model_variant = Some(model_variant);
        self
    }

    pub fn with_phase(mut self, phase: RuntimeTracePhase) -> Self {
        self.phase = phase;
        self
    }

    pub const fn contains_payload(&self) -> bool {
        false
    }

    pub const fn payload_redacted(&self) -> bool {
        matches!(self.redaction, ReplayRedaction::SanitizedMetadataOnly)
    }
}

pub const RUNTIME_REPLAY_REDACTION: ReplayRedaction =
    ReplayRedaction::SanitizedMetadataOnly;

pub fn sanitized_replay_record(
    request_id: impl Into<String>,
    capability: ConformanceCapability,
    model_variant: Option<ModelVariant>,
    correlation_id: Option<String>,
    phase: RuntimeTracePhase,
) -> RuntimeReplayRecord {
    let mut record = RuntimeReplayRecord::sanitized(request_id, capability).with_phase(phase);

    if let Some(model_variant) = model_variant {
        record = record.with_model_variant(model_variant);
    }
    if let Some(correlation_id) = correlation_id {
        record = record.with_correlation_id(correlation_id);
    }

    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_record_is_sanitized_metadata_only() {
        let record = RuntimeReplayRecord::sanitized(
            "req-1",
            ConformanceCapability::Chat,
        )
        .with_correlation_id("corr-1")
        .with_model_variant(ModelVariant::Qwen38BGguf)
        .with_phase(RuntimeTracePhase::Decode);

        assert_eq!(record.request_id, "req-1");
        assert_eq!(record.correlation_id.as_deref(), Some("corr-1"));
        assert_eq!(record.capability, ConformanceCapability::Chat);
        assert_eq!(record.model_variant, Some(ModelVariant::Qwen38BGguf));
        assert_eq!(record.phase, RuntimeTracePhase::Decode);
        assert_eq!(record.redaction, ReplayRedaction::SanitizedMetadataOnly);
        assert!(record.payload_redacted());
        assert!(!record.contains_payload());
    }

    #[test]
    fn replay_helper_preserves_routing_metadata_without_payloads() {
        let record = sanitized_replay_record(
            "req-audio",
            ConformanceCapability::SpeechToSpeech,
            Some(ModelVariant::Lfm25Audio15BGguf),
            Some("corr-audio".to_string()),
            RuntimeTracePhase::FirstChunk,
        );

        assert_eq!(record.request_id, "req-audio");
        assert_eq!(record.correlation_id.as_deref(), Some("corr-audio"));
        assert_eq!(
            record.capability,
            ConformanceCapability::SpeechToSpeech
        );
        assert_eq!(
            record.model_variant,
            Some(ModelVariant::Lfm25Audio15BGguf)
        );
        assert_eq!(record.phase, RuntimeTracePhase::FirstChunk);
        assert!(!record.contains_payload());
    }
}

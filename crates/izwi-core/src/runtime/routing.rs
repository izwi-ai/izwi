//! Runtime routing decision contracts.
//!
//! This layer validates route intent against capability adapters and backend
//! selection before any execution path is cut over to broker-controlled routing.

use serde::Serialize;

use crate::backends::{BackendKind, BackendRouter};
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::runtime::adapters::{CapabilityKind, RuntimeAdapterRegistry};
use crate::runtime::capabilities::{
    CapabilityExecutionPlan, CapabilityExecutionRegistry, CapabilityExecutionRequest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RouteSource {
    OpenAiChatCompletions,
    OpenAiResponses,
    OpenAiAudioSpeech,
    OpenAiAudioTranscriptions,
    ProductTranscriptionJob,
    ProductSpeechJob,
    ProductDiarizationRecord,
    ProductSavedVoice,
    ProductVoiceSession,
    RealtimeTranscription,
    RealtimeVoice,
    AdminModels,
    InternalEngine,
    InternalRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RoutingInputKind {
    Text,
    Audio,
    AudioStream,
    TextAndAudio,
    VoiceReference,
    Artifact,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LatencyClass {
    Batch,
    Interactive,
    Realtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QualityClass {
    Default,
    Draft,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CostClass {
    Default,
    Economy,
    Unbounded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct RoutingCandidateTrace {
    pub(crate) model_variant: ModelVariant,
    pub(crate) accepted: bool,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct RoutingRequest {
    pub(crate) source: RouteSource,
    pub(crate) capability: CapabilityKind,
    pub(crate) input_kind: RoutingInputKind,
    pub(crate) requested_model_variant: Option<ModelVariant>,
    pub(crate) streaming_required: bool,
    pub(crate) latency_class: LatencyClass,
    pub(crate) quality_class: QualityClass,
    pub(crate) cost_class: CostClass,
    pub(crate) allow_model_fallback: bool,
    pub(crate) candidate_models: Vec<ModelVariant>,
}

impl RoutingRequest {
    pub(crate) fn new(source: RouteSource, capability: CapabilityKind) -> Self {
        Self {
            source,
            capability,
            input_kind: RoutingInputKind::Unknown,
            requested_model_variant: None,
            streaming_required: false,
            latency_class: LatencyClass::Interactive,
            quality_class: QualityClass::Default,
            cost_class: CostClass::Default,
            allow_model_fallback: false,
            candidate_models: Vec::new(),
        }
    }

    pub(crate) fn with_input_kind(mut self, input_kind: RoutingInputKind) -> Self {
        self.input_kind = input_kind;
        self
    }

    pub(crate) fn with_model_variant(mut self, model_variant: ModelVariant) -> Self {
        self.requested_model_variant = Some(model_variant);
        self
    }

    pub(crate) fn with_optional_model_variant(
        mut self,
        model_variant: Option<ModelVariant>,
    ) -> Self {
        self.requested_model_variant = model_variant;
        self
    }

    pub(crate) fn with_streaming_required(mut self, streaming_required: bool) -> Self {
        self.streaming_required = streaming_required;
        self
    }

    pub(crate) fn with_latency_class(mut self, latency_class: LatencyClass) -> Self {
        self.latency_class = latency_class;
        self
    }

    pub(crate) fn with_quality_class(mut self, quality_class: QualityClass) -> Self {
        self.quality_class = quality_class;
        self
    }

    pub(crate) fn with_cost_class(mut self, cost_class: CostClass) -> Self {
        self.cost_class = cost_class;
        self
    }

    pub(crate) fn with_model_fallback(mut self, allow_model_fallback: bool) -> Self {
        self.allow_model_fallback = allow_model_fallback;
        self
    }

    pub(crate) fn with_candidate_models(mut self, candidate_models: Vec<ModelVariant>) -> Self {
        self.candidate_models = candidate_models;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct RoutingDecision {
    pub(crate) source: RouteSource,
    pub(crate) capability: CapabilityKind,
    pub(crate) input_kind: RoutingInputKind,
    pub(crate) requested_model_variant: Option<ModelVariant>,
    pub(crate) selected_model_variant: ModelVariant,
    pub(crate) execution_plan: CapabilityExecutionPlan,
    pub(crate) backend_kind: BackendKind,
    pub(crate) backend_reason: String,
    pub(crate) backend_diagnostics: Vec<String>,
    pub(crate) latency_class: LatencyClass,
    pub(crate) quality_class: QualityClass,
    pub(crate) cost_class: CostClass,
    pub(crate) fallback_chain: Vec<RoutingCandidateTrace>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RuntimeRouter<'a> {
    adapters: &'a RuntimeAdapterRegistry,
    backend_router: &'a BackendRouter,
}

impl<'a> RuntimeRouter<'a> {
    pub(crate) fn new(
        adapters: &'a RuntimeAdapterRegistry,
        backend_router: &'a BackendRouter,
    ) -> Self {
        Self {
            adapters,
            backend_router,
        }
    }

    pub(crate) fn plan(&self, request: RoutingRequest) -> Result<RoutingDecision> {
        let candidates = self.candidates_for(&request)?;
        let execution_registry = CapabilityExecutionRegistry::new(self.adapters);
        let mut fallback_chain = Vec::new();

        for model_variant in candidates {
            let backend_plan = self.backend_router.select(model_variant);
            let execution_request = CapabilityExecutionRequest::new(
                request.capability,
                model_variant,
                backend_plan.backend.kind(),
            )
            .with_streaming_required(request.streaming_required);

            match execution_registry.plan(execution_request) {
                Ok(execution_plan) => {
                    fallback_chain.push(RoutingCandidateTrace {
                        model_variant,
                        accepted: true,
                        reason: format!(
                            "selected adapter {} with {:?} execution",
                            execution_plan.adapter_id, execution_plan.execution_target
                        ),
                    });

                    return Ok(RoutingDecision {
                        source: request.source,
                        capability: request.capability,
                        input_kind: request.input_kind,
                        requested_model_variant: request.requested_model_variant,
                        selected_model_variant: model_variant,
                        execution_plan,
                        backend_kind: backend_plan.backend.kind(),
                        backend_reason: backend_plan.reason,
                        backend_diagnostics: backend_plan.diagnostics,
                        latency_class: request.latency_class,
                        quality_class: request.quality_class,
                        cost_class: request.cost_class,
                        fallback_chain,
                    });
                }
                Err(err) => fallback_chain.push(RoutingCandidateTrace {
                    model_variant,
                    accepted: false,
                    reason: err.to_string(),
                }),
            }
        }

        Err(Error::InvalidInput(format!(
            "No route found for {:?} from {:?}: {}",
            request.capability,
            request.source,
            fallback_chain
                .iter()
                .map(|candidate| format!("{} ({})", candidate.model_variant, candidate.reason))
                .collect::<Vec<_>>()
                .join("; ")
        )))
    }

    fn candidates_for(&self, request: &RoutingRequest) -> Result<Vec<ModelVariant>> {
        if let Some(model_variant) = request.requested_model_variant {
            return Ok(vec![model_variant]);
        }

        if !request.allow_model_fallback {
            return Err(Error::InvalidInput(format!(
                "Runtime router could not validate {:?} from {:?}: request missing model variant",
                request.capability, request.source
            )));
        }

        if !request.candidate_models.is_empty() {
            return Ok(request.candidate_models.clone());
        }

        Ok(ModelVariant::all().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::BackendPreference;
    use crate::runtime::adapters::{ExecutionTargetKind, StreamingMode};

    fn router_fixture() -> (RuntimeAdapterRegistry, BackendRouter) {
        (
            RuntimeAdapterRegistry::built_in(),
            BackendRouter::from_preference(BackendPreference::Cpu),
        )
    }

    #[test]
    fn router_plans_explicit_supported_model() {
        let (adapters, backend_router) = router_fixture();
        let router = RuntimeRouter::new(&adapters, &backend_router);

        let decision = router
            .plan(
                RoutingRequest::new(RouteSource::OpenAiChatCompletions, CapabilityKind::Chat)
                    .with_input_kind(RoutingInputKind::Text)
                    .with_model_variant(ModelVariant::Qwen38BGguf),
            )
            .expect("route decision");

        assert_eq!(decision.selected_model_variant, ModelVariant::Qwen38BGguf);
        assert_eq!(decision.backend_kind, BackendKind::Cpu);
        assert_eq!(
            decision.execution_plan.execution_target,
            ExecutionTargetKind::TokenEngine
        );
        assert_eq!(decision.fallback_chain.len(), 1);
        assert!(decision.fallback_chain[0].accepted);
    }

    #[test]
    fn router_rejects_missing_model_without_fallback() {
        let (adapters, backend_router) = router_fixture();
        let router = RuntimeRouter::new(&adapters, &backend_router);

        let err = router
            .plan(RoutingRequest::new(
                RouteSource::InternalEngine,
                CapabilityKind::Tts,
            ))
            .unwrap_err();

        assert!(err.to_string().contains("missing model variant"));
    }

    #[test]
    fn router_records_fallback_chain_until_candidate_matches() {
        let (adapters, backend_router) = router_fixture();
        let router = RuntimeRouter::new(&adapters, &backend_router);

        let decision = router
            .plan(
                RoutingRequest::new(RouteSource::OpenAiChatCompletions, CapabilityKind::Chat)
                    .with_model_fallback(true)
                    .with_candidate_models(vec![
                        ModelVariant::Kokoro82M,
                        ModelVariant::Qwen38BGguf,
                    ]),
            )
            .expect("fallback route decision");

        assert_eq!(decision.selected_model_variant, ModelVariant::Qwen38BGguf);
        assert_eq!(decision.fallback_chain.len(), 2);
        assert!(!decision.fallback_chain[0].accepted);
        assert!(decision.fallback_chain[0]
            .reason
            .contains("does not support"));
        assert!(decision.fallback_chain[1].accepted);
    }

    #[test]
    fn router_respects_streaming_requirement_across_candidates() {
        let (adapters, backend_router) = router_fixture();
        let router = RuntimeRouter::new(&adapters, &backend_router);

        let decision = router
            .plan(
                RoutingRequest::new(RouteSource::RealtimeTranscription, CapabilityKind::Asr)
                    .with_input_kind(RoutingInputKind::AudioStream)
                    .with_latency_class(LatencyClass::Realtime)
                    .with_model_fallback(true)
                    .with_streaming_required(true)
                    .with_candidate_models(vec![
                        ModelVariant::WhisperLargeV3Turbo,
                        ModelVariant::Lfm25Audio15BGguf,
                    ]),
            )
            .expect("streaming ASR fallback decision");

        assert_eq!(
            decision.selected_model_variant,
            ModelVariant::Lfm25Audio15BGguf
        );
        assert_eq!(
            decision.execution_plan.streaming_mode,
            StreamingMode::Chunked
        );
        assert!(!decision.fallback_chain[0].accepted);
        assert!(decision.fallback_chain[1].accepted);
    }

    #[test]
    fn router_returns_candidate_errors_when_no_model_matches() {
        let (adapters, backend_router) = router_fixture();
        let router = RuntimeRouter::new(&adapters, &backend_router);

        let err = router
            .plan(
                RoutingRequest::new(RouteSource::OpenAiAudioSpeech, CapabilityKind::Tts)
                    .with_model_fallback(true)
                    .with_candidate_models(vec![
                        ModelVariant::Qwen38BGguf,
                        ModelVariant::WhisperLargeV3Turbo,
                    ]),
            )
            .unwrap_err();

        assert!(err.to_string().contains("No route found"));
        assert!(err
            .to_string()
            .contains(ModelVariant::Qwen38BGguf.display_name()));
        assert!(err
            .to_string()
            .contains(ModelVariant::WhisperLargeV3Turbo.display_name()));
    }
}

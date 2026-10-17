//! Rollout-aware inference broker skeleton.
//!
//! Phase 3 keeps this deliberately inert: the broker can be configured and
//! inspected, but runtime execution still follows existing `RuntimeService`
//! paths until later phases explicitly opt capabilities in.

use serde::Serialize;

use crate::backends::BackendRouter;
use crate::engine::{EngineCoreRequest, TaskType};
use crate::runtime::adapters::{CapabilityKind, RuntimeAdapterRegistry};
use crate::runtime::routing::{RouteSource, RoutingDecision, RoutingRequest, RuntimeRouter};

const BROKER_MODE_ENV: &str = "IZWI_INFERENCE_BROKER";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InferenceBrokerMode {
    Off,
    Shadow,
    On,
}

impl InferenceBrokerMode {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "shadow" | "audit" => Self::Shadow,
            "on" | "enabled" | "true" | "1" => Self::On,
            _ => Self::Off,
        }
    }

    fn from_env() -> Self {
        std::env::var(BROKER_MODE_ENV)
            .ok()
            .as_deref()
            .map(Self::parse)
            .unwrap_or(Self::Off)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InferenceBrokerObservation {
    pub(crate) source: RouteSource,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: Option<crate::model::ModelVariant>,
    pub(crate) routing_decision: Option<RoutingDecision>,
    pub(crate) shadow_enabled: bool,
    pub(crate) execution_enabled: bool,
    pub(crate) validation_error: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceBroker {
    mode: InferenceBrokerMode,
}

impl InferenceBroker {
    pub(crate) fn from_env() -> Self {
        Self {
            mode: InferenceBrokerMode::from_env(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_mode(mode: InferenceBrokerMode) -> Self {
        Self { mode }
    }

    pub(crate) fn shadow_enabled(&self) -> bool {
        matches!(
            self.mode,
            InferenceBrokerMode::Shadow | InferenceBrokerMode::On
        )
    }

    pub(crate) fn execution_enabled(&self) -> bool {
        matches!(self.mode, InferenceBrokerMode::On)
    }

    #[cfg(test)]
    pub(crate) fn observe_engine_request(
        &self,
        request: &EngineCoreRequest,
        adapters: &RuntimeAdapterRegistry,
        backend_router: &BackendRouter,
    ) -> Option<InferenceBrokerObservation> {
        self.observe_engine_request_with_streaming_required(
            request,
            request.streaming,
            adapters,
            backend_router,
        )
    }

    pub(crate) fn observe_engine_request_with_streaming_required(
        &self,
        request: &EngineCoreRequest,
        streaming_required: bool,
        adapters: &RuntimeAdapterRegistry,
        backend_router: &BackendRouter,
    ) -> Option<InferenceBrokerObservation> {
        self.observe_capability_request(
            RouteSource::InternalEngine,
            capability_for_task(request.task_type),
            request.model_variant,
            streaming_required,
            adapters,
            backend_router,
        )
    }

    pub(crate) fn observe_capability_request(
        &self,
        source: RouteSource,
        capability: CapabilityKind,
        model_variant: Option<crate::model::ModelVariant>,
        streaming_required: bool,
        adapters: &RuntimeAdapterRegistry,
        backend_router: &BackendRouter,
    ) -> Option<InferenceBrokerObservation> {
        if !self.shadow_enabled() && !self.execution_enabled() {
            return None;
        }

        let router = RuntimeRouter::new(adapters, backend_router);
        let route_request = RoutingRequest::new(source, capability)
            .with_optional_model_variant(model_variant)
            .with_streaming_required(streaming_required);
        let (routing_decision, validation_error) = match router.plan(route_request) {
            Ok(decision) => (Some(decision), None),
            Err(err) => (None, Some(err.to_string())),
        };

        Some(InferenceBrokerObservation {
            source,
            capability,
            model_variant,
            routing_decision,
            shadow_enabled: self.shadow_enabled(),
            execution_enabled: self.execution_enabled(),
            validation_error,
        })
    }
}

fn capability_for_task(task_type: TaskType) -> CapabilityKind {
    CapabilityKind::for_engine_task(task_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendPreference, BackendRouter};
    use crate::engine::EngineCoreRequest;
    use crate::model::ModelVariant;
    use crate::runtime::adapters::RuntimeAdapterRegistry;

    fn route_test_fixture() -> (RuntimeAdapterRegistry, BackendRouter) {
        (
            RuntimeAdapterRegistry::built_in(),
            BackendRouter::from_preference(BackendPreference::Cpu),
        )
    }

    #[test]
    fn broker_mode_defaults_unknown_values_to_off() {
        assert_eq!(InferenceBrokerMode::parse(""), InferenceBrokerMode::Off);
        assert_eq!(
            InferenceBrokerMode::parse("something-else"),
            InferenceBrokerMode::Off
        );
    }

    #[test]
    fn broker_mode_accepts_shadow_alias() {
        assert_eq!(
            InferenceBrokerMode::parse("shadow"),
            InferenceBrokerMode::Shadow
        );
        assert_eq!(
            InferenceBrokerMode::parse("audit"),
            InferenceBrokerMode::Shadow
        );
    }

    #[test]
    fn broker_modes_separate_shadow_from_execution() {
        let off = InferenceBroker::with_mode(InferenceBrokerMode::Off);
        assert!(!off.shadow_enabled());
        assert!(!off.execution_enabled());

        let shadow = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
        assert!(shadow.shadow_enabled());
        assert!(!shadow.execution_enabled());

        let on = InferenceBroker::with_mode(InferenceBrokerMode::On);
        assert!(on.shadow_enabled());
        assert!(on.execution_enabled());
    }

    #[test]
    fn broker_shadow_observes_and_validates_engine_requests() {
        let broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
        let (adapters, backend_router) = route_test_fixture();
        let request = EngineCoreRequest::chat(vec![]).with_model_variant(ModelVariant::Qwen359BGguf);

        let observation = broker
            .observe_engine_request(&request, &adapters, &backend_router)
            .expect("shadow mode should observe request");

        assert_eq!(observation.source, RouteSource::InternalEngine);
        assert_eq!(observation.capability, CapabilityKind::Chat);
        assert!(observation.shadow_enabled);
        assert!(!observation.execution_enabled);
        assert!(observation.routing_decision.is_some());
        assert!(observation.validation_error.is_none());
    }

    #[test]
    fn broker_can_validate_transport_streaming_as_offline_execution() {
        let broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
        let (adapters, backend_router) = route_test_fixture();
        let mut request =
            EngineCoreRequest::asr("audio").with_model_variant(ModelVariant::ParakeetTdt06BV3);
        request.streaming = true;

        let observation = broker
            .observe_engine_request_with_streaming_required(
                &request,
                false,
                &adapters,
                &backend_router,
            )
            .expect("shadow mode should observe transport streaming request");

        assert_eq!(observation.source, RouteSource::InternalEngine);
        assert_eq!(observation.capability, CapabilityKind::Asr);
        assert!(observation.routing_decision.is_some());
        assert!(observation.validation_error.is_none());
    }

    #[test]
    fn broker_on_reports_adapter_validation_errors() {
        let broker = InferenceBroker::with_mode(InferenceBrokerMode::On);
        let (adapters, backend_router) = route_test_fixture();
        let request = EngineCoreRequest::chat(vec![]).with_model_variant(ModelVariant::Kokoro82M);

        let observation = broker
            .observe_engine_request(&request, &adapters, &backend_router)
            .expect("on mode should observe request");

        assert!(observation.execution_enabled);
        assert!(observation.routing_decision.is_none());
        assert!(observation.validation_error.is_some());
    }

    #[test]
    fn broker_observes_direct_capability_requests() {
        let broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
        let (adapters, backend_router) = route_test_fixture();

        let observation = broker
            .observe_capability_request(
                RouteSource::InternalRuntime,
                CapabilityKind::Tts,
                Some(ModelVariant::Kokoro82M),
                true,
                &adapters,
                &backend_router,
            )
            .expect("shadow broker should observe direct TTS request");

        assert_eq!(observation.source, RouteSource::InternalRuntime);
        assert_eq!(observation.capability, CapabilityKind::Tts);
        assert_eq!(observation.model_variant, Some(ModelVariant::Kokoro82M));
        assert!(observation.shadow_enabled);
        assert!(!observation.execution_enabled);
        assert!(observation.routing_decision.is_some());
        assert!(observation.validation_error.is_none());
    }
}

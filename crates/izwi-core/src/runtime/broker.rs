//! Rollout-aware inference broker skeleton.
//!
//! Phase 3 keeps this deliberately inert: the broker can be configured and
//! inspected, but runtime execution still follows existing `RuntimeService`
//! paths until later phases explicitly opt capabilities in.

use serde::Serialize;

use crate::engine::{EngineCoreRequest, TaskType};
use crate::runtime::adapters::{CapabilityKind, RuntimeAdapterRegistry};

const BROKER_MODE_ENV: &str = "IZWI_INFERENCE_BROKER";
const DEPLOYMENT_MODE_ENV: &str = "IZWI_INFERENCE_DEPLOYMENT_MODE";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InferenceDeploymentMode {
    Local,
    Gateway,
    Worker,
}

impl InferenceDeploymentMode {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "gateway" | "router" => Self::Gateway,
            "worker" => Self::Worker,
            _ => Self::Local,
        }
    }

    fn from_env() -> Self {
        std::env::var(DEPLOYMENT_MODE_ENV)
            .ok()
            .as_deref()
            .map(Self::parse)
            .unwrap_or(Self::Local)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct InferenceBrokerSnapshot {
    pub(crate) mode: InferenceBrokerMode,
    pub(crate) deployment_mode: InferenceDeploymentMode,
    pub(crate) shadow_enabled: bool,
    pub(crate) execution_enabled: bool,
    pub(crate) local_runtime_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InferenceBrokerObservation {
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: Option<crate::model::ModelVariant>,
    pub(crate) shadow_enabled: bool,
    pub(crate) execution_enabled: bool,
    pub(crate) validation_error: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceBroker {
    mode: InferenceBrokerMode,
    deployment_mode: InferenceDeploymentMode,
}

impl InferenceBroker {
    pub(crate) fn from_env() -> Self {
        Self {
            mode: InferenceBrokerMode::from_env(),
            deployment_mode: InferenceDeploymentMode::from_env(),
        }
    }

    pub(crate) fn with_mode(mode: InferenceBrokerMode) -> Self {
        Self {
            mode,
            deployment_mode: InferenceDeploymentMode::Local,
        }
    }

    pub(crate) fn with_mode_and_deployment(
        mode: InferenceBrokerMode,
        deployment_mode: InferenceDeploymentMode,
    ) -> Self {
        Self {
            mode,
            deployment_mode,
        }
    }

    pub(crate) fn mode(&self) -> InferenceBrokerMode {
        self.mode
    }

    pub(crate) fn deployment_mode(&self) -> InferenceDeploymentMode {
        self.deployment_mode
    }

    pub(crate) fn local_runtime_default(&self) -> bool {
        matches!(self.deployment_mode, InferenceDeploymentMode::Local)
    }

    pub(crate) fn shadow_enabled(&self) -> bool {
        matches!(self.mode, InferenceBrokerMode::Shadow | InferenceBrokerMode::On)
    }

    pub(crate) fn execution_enabled(&self) -> bool {
        matches!(self.mode, InferenceBrokerMode::On)
    }

    pub(crate) fn snapshot(&self) -> InferenceBrokerSnapshot {
        InferenceBrokerSnapshot {
            mode: self.mode(),
            deployment_mode: self.deployment_mode(),
            shadow_enabled: self.shadow_enabled(),
            execution_enabled: self.execution_enabled(),
            local_runtime_default: self.local_runtime_default(),
        }
    }

    pub(crate) fn observe_engine_request(
        &self,
        request: &EngineCoreRequest,
        adapters: &RuntimeAdapterRegistry,
    ) -> Option<InferenceBrokerObservation> {
        if !self.shadow_enabled() && !self.execution_enabled() {
            return None;
        }

        let capability = capability_for_task(request.task_type);
        let validation_error = match request.model_variant {
            Some(model_variant) => adapters
                .require(capability, model_variant)
                .err()
                .map(|err| err.to_string()),
            None => Some(format!(
                "Inference broker could not validate {capability:?}: request missing model variant"
            )),
        };

        Some(InferenceBrokerObservation {
            capability,
            model_variant: request.model_variant,
            shadow_enabled: self.shadow_enabled(),
            execution_enabled: self.execution_enabled(),
            validation_error,
        })
    }
}

fn capability_for_task(task_type: TaskType) -> CapabilityKind {
    match task_type {
        TaskType::TTS => CapabilityKind::Tts,
        TaskType::ASR => CapabilityKind::Asr,
        TaskType::Chat => CapabilityKind::Chat,
        TaskType::SpeechToSpeech => CapabilityKind::SpeechToSpeech,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineCoreRequest;
    use crate::model::ModelVariant;
    use crate::runtime::adapters::RuntimeAdapterRegistry;

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
        assert_eq!(InferenceBrokerMode::parse("shadow"), InferenceBrokerMode::Shadow);
        assert_eq!(InferenceBrokerMode::parse("audit"), InferenceBrokerMode::Shadow);
    }

    #[test]
    fn deployment_mode_defaults_to_local_for_unknown_values() {
        assert_eq!(InferenceDeploymentMode::parse(""), InferenceDeploymentMode::Local);
        assert_eq!(
            InferenceDeploymentMode::parse("something-else"),
            InferenceDeploymentMode::Local
        );
    }

    #[test]
    fn deployment_mode_requires_explicit_gateway_or_worker() {
        assert_eq!(
            InferenceDeploymentMode::parse("gateway"),
            InferenceDeploymentMode::Gateway
        );
        assert_eq!(
            InferenceDeploymentMode::parse("router"),
            InferenceDeploymentMode::Gateway
        );
        assert_eq!(
            InferenceDeploymentMode::parse("worker"),
            InferenceDeploymentMode::Worker
        );
    }

    #[test]
    fn broker_snapshot_separates_shadow_from_execution() {
        let off = InferenceBroker::with_mode(InferenceBrokerMode::Off).snapshot();
        assert!(!off.shadow_enabled);
        assert!(!off.execution_enabled);

        let shadow = InferenceBroker::with_mode(InferenceBrokerMode::Shadow).snapshot();
        assert!(shadow.shadow_enabled);
        assert!(!shadow.execution_enabled);

        let on = InferenceBroker::with_mode(InferenceBrokerMode::On).snapshot();
        assert!(on.shadow_enabled);
        assert!(on.execution_enabled);
    }

    #[test]
    fn broker_snapshot_keeps_local_runtime_as_default() {
        let local = InferenceBroker::with_mode(InferenceBrokerMode::Off).snapshot();
        assert_eq!(local.deployment_mode, InferenceDeploymentMode::Local);
        assert!(local.local_runtime_default);

        let gateway = InferenceBroker::with_mode_and_deployment(
            InferenceBrokerMode::Shadow,
            InferenceDeploymentMode::Gateway,
        )
        .snapshot();
        assert_eq!(gateway.deployment_mode, InferenceDeploymentMode::Gateway);
        assert!(!gateway.local_runtime_default);
        assert!(gateway.shadow_enabled);
    }

    #[test]
    fn broker_shadow_observes_and_validates_engine_requests() {
        let broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
        let adapters = RuntimeAdapterRegistry::built_in();
        let request = EngineCoreRequest::chat(vec![]).with_model_variant(ModelVariant::Qwen38BGguf);

        let observation = broker
            .observe_engine_request(&request, &adapters)
            .expect("shadow mode should observe request");

        assert_eq!(observation.capability, CapabilityKind::Chat);
        assert!(observation.shadow_enabled);
        assert!(!observation.execution_enabled);
        assert!(observation.validation_error.is_none());
    }

    #[test]
    fn broker_on_reports_adapter_validation_errors() {
        let broker = InferenceBroker::with_mode(InferenceBrokerMode::On);
        let adapters = RuntimeAdapterRegistry::built_in();
        let request = EngineCoreRequest::chat(vec![]).with_model_variant(ModelVariant::Kokoro82M);

        let observation = broker
            .observe_engine_request(&request, &adapters)
            .expect("on mode should observe request");

        assert!(observation.execution_enabled);
        assert!(observation.validation_error.is_some());
    }
}

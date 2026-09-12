//! Multi-worker routing for the migrated public chat slice.
//!
//! The registry narrows candidates using fresh, receiver-clock status, but the
//! chosen worker remains authoritative for admission. Each dispatch performs
//! exactly one invocation POST. Explicit rejection, uncertain acceptance, and
//! accepted execution are all returned from that worker without failover.

use std::time::Duration;

use izwi_core::{ChatGeneration, ModelVariant};
use izwi_serving_protocol::{
    CancellationBehavior, DeploymentId, InputFormat, ModelAlias, OutputFormat, PolicyRevision,
    TaskKind, PROTOCOL_V1,
};
use tokio::sync::mpsc;

use super::chat::{
    generate_remote_chat_with_execution_and_acceptance, spawn_remote_chat_stream_with_execution,
    ChatExecutionRequest, ChatStreamEvent, RemoteChatExecution, RemoteChatExecutionConfig,
};
use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::worker_registry::{
    BackendPolicy, SelectedWorker, WorkerRegistry, WorkerRegistryError, WorkerSelectionRequest,
};

const FORWARDED_CHAT_STREAM_CAPACITY: usize = 64;

#[derive(Debug, Clone)]
pub struct RemoteChatDispatchConfig {
    pub public_model_variant: ModelVariant,
    pub deployment_id: DeploymentId,
    pub policy_revision: PolicyRevision,
    pub backend_policy: BackendPolicy,
    pub max_queue_wait: Duration,
    pub max_output_tokens: u32,
    pub max_output_bytes: u64,
}

impl RemoteChatDispatchConfig {
    fn validate(&self) -> Result<(), ApiError> {
        if self.max_output_tokens == 0 || self.max_output_bytes == 0 {
            return Err(ApiError::internal(
                "Remote chat dispatch output limits must be non-zero",
            ));
        }
        ModelAlias::new(self.public_model_variant.dir_name()).map_err(|error| {
            ApiError::internal(format!("Invalid public chat model alias: {error}"))
        })?;
        Ok(())
    }
}

/// Bounded, hardware-independent dispatcher for one migrated public chat
/// deployment. Adding workers changes capacity, not the public API contract.
#[derive(Debug, Clone)]
pub struct RemoteChatDispatcher {
    registry: WorkerRegistry,
    config: RemoteChatDispatchConfig,
}

impl RemoteChatDispatcher {
    pub fn new(
        registry: WorkerRegistry,
        config: RemoteChatDispatchConfig,
    ) -> Result<Self, ApiError> {
        config.validate()?;
        Ok(Self { registry, config })
    }

    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    /// Readiness is derived from a receiver-clock-fresh registry observation,
    /// never from registration alone. Exhausted workers remain ready because
    /// capacity affects admission, not service health.
    pub fn readiness_check(&self) -> Result<(), String> {
        let public_model = ModelAlias::new(self.config.public_model_variant.dir_name())
            .map_err(|error| format!("invalid configured public model alias: {error}"))?;
        let compatible = [false, true].into_iter().all(|streaming| {
            self.registry
                .has_fresh_compatible_worker(&WorkerSelectionRequest {
                    protocol_version: PROTOCOL_V1,
                    deployment_id: self.config.deployment_id.clone(),
                    public_model: public_model.clone(),
                    task: TaskKind::Chat,
                    input_format: InputFormat::ChatMessages,
                    output_format: OutputFormat::Text,
                    streaming,
                    realtime: false,
                    cancellation: Some(CancellationBehavior::Cooperative),
                    backend_policy: self.config.backend_policy,
                    input_bytes: 0,
                    context_tokens: None,
                    output_tokens: Some(1),
                })
        });
        if !compatible {
            return Err(format!(
                "no fresh compatible worker is ready for deployment {}",
                self.config.deployment_id
            ));
        }
        Ok(())
    }

    pub async fn generate(
        &self,
        request_timeout_secs: u64,
        context: &RequestContext,
        request: ChatExecutionRequest,
    ) -> Result<ChatGeneration, ApiError> {
        let selected = self.select(&request, false)?;
        let remote = self.execution_for(&selected)?;
        generate_remote_chat_with_execution_and_acceptance(
            &remote,
            request_timeout_secs,
            context,
            request,
            selected.dispatch,
            |dispatch| dispatch.mark_accepted().map_err(map_registry_error),
        )
        .await
    }

    /// Returns only after the selected worker has emitted a contract-valid
    /// accepted event. The forwarding task owns the local dispatch reservation
    /// through terminal delivery or public-consumer disconnect.
    pub async fn stream(
        &self,
        request_timeout_secs: u64,
        context: &RequestContext,
        request: ChatExecutionRequest,
    ) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError> {
        let mut selected = self.select(&request, true)?;
        let remote = self.execution_for(&selected)?;
        let mut worker_events = spawn_remote_chat_stream_with_execution(
            &remote,
            request_timeout_secs,
            context,
            request,
        )
        .await?;
        selected
            .dispatch
            .mark_accepted()
            .map_err(map_registry_error)?;

        let (public_tx, public_rx) = mpsc::channel(FORWARDED_CHAT_STREAM_CAPACITY);
        tokio::spawn(async move {
            // Keep the reservation alive while the private stream is live.
            // Dropping either receiver propagates cancellation toward the
            // exact accepted attempt; there is intentionally no reselection.
            let _dispatch = selected.dispatch;
            while let Some(event) = worker_events.recv().await {
                let terminal = matches!(
                    event,
                    ChatStreamEvent::Completed(_)
                        | ChatStreamEvent::Failed(_)
                        | ChatStreamEvent::ShuttingDown
                );
                if public_tx.send(event).await.is_err() || terminal {
                    break;
                }
            }
        });
        Ok(public_rx)
    }

    fn select(
        &self,
        request: &ChatExecutionRequest,
        streaming: bool,
    ) -> Result<SelectedWorker, ApiError> {
        if request.variant != self.config.public_model_variant {
            return Err(ApiError::bad_request(format!(
                "Requested model is incompatible with remote deployment {}",
                self.config.deployment_id
            )));
        }

        let input_bytes = request.messages.iter().fold(0u64, |total, message| {
            total.saturating_add(u64::try_from(message.content.len()).unwrap_or(u64::MAX))
        });
        let requested_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(usize::MAX)
            .max(1);
        let output_tokens = u32::try_from(requested_tokens)
            .unwrap_or(u32::MAX)
            .min(self.config.max_output_tokens)
            .max(1);
        let public_model = ModelAlias::new(self.config.public_model_variant.dir_name())
            .map_err(|error| ApiError::internal(format!("Invalid public model alias: {error}")))?;
        let selection = WorkerSelectionRequest {
            protocol_version: PROTOCOL_V1,
            deployment_id: self.config.deployment_id.clone(),
            public_model,
            task: TaskKind::Chat,
            input_format: InputFormat::ChatMessages,
            output_format: OutputFormat::Text,
            streaming,
            realtime: false,
            cancellation: Some(CancellationBehavior::Cooperative),
            backend_policy: self.config.backend_policy,
            input_bytes,
            // Exact prompt tokenization belongs to the selected worker. Input
            // bytes are bounded here and context limits are enforced there.
            context_tokens: None,
            output_tokens: Some(output_tokens),
        };
        self.registry
            .select_and_reserve(&selection)
            .map_err(map_registry_error)
    }

    fn execution_for(&self, selected: &SelectedWorker) -> Result<RemoteChatExecution, ApiError> {
        RemoteChatExecution::new(
            selected.client.clone(),
            RemoteChatExecutionConfig {
                public_model_variant: self.config.public_model_variant,
                expected_worker_incarnation: selected.key.incarnation_id.clone(),
                deployment_id: selected.deployment_id.clone(),
                expected_model_generation: selected.model_generation,
                policy_revision: self.config.policy_revision.clone(),
                max_queue_wait: self.config.max_queue_wait,
                max_output_tokens: self.config.max_output_tokens,
                max_output_bytes: self.config.max_output_bytes,
            },
        )
    }
}

fn map_registry_error(error: WorkerRegistryError) -> ApiError {
    match error {
        WorkerRegistryError::NoEligibleWorker | WorkerRegistryError::LocalDispatchLimitReached => {
            ApiError::service_unavailable(error.to_string())
        }
        _ => ApiError::internal(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    use izwi_core::{ChatMessage, ChatRequestConfig, ChatRole};
    use izwi_hooks::Principal;
    use izwi_serving_client::{
        mock::{MockFault, MockWorker, MockWorkerConfig},
        WorkerClient, WorkerClientConfig,
    };
    use izwi_serving_protocol::{
        ArtifactRevision, BackendKind, Capability, CapacitySnapshot, CredentialId,
        DeviceAssignment, LoadedDeployment, ModelGeneration, ModelReadiness, NodeId,
        ServiceBearerToken, ServiceCredentials, WorkerDescriptor, WorkerFeature, WorkerId,
        WorkerProcessState, WorkerStatus,
    };

    use crate::worker_registry::{ApprovedDeployment, ApprovedWorker, WorkerRegistryConfig};

    fn id<T>(value: &str) -> T
    where
        T: TryFrom<String>,
        T::Error: std::fmt::Debug,
    {
        T::try_from(value.to_string()).unwrap()
    }

    fn credentials() -> ServiceCredentials {
        ServiceCredentials {
            credential_id: id::<CredentialId>("gateway-test-key"),
            bearer_token: ServiceBearerToken::new("gateway-test-secret").unwrap(),
        }
    }

    fn client(endpoint: &str) -> WorkerClient {
        WorkerClient::new(endpoint, credentials(), WorkerClientConfig::default()).unwrap()
    }

    fn deployment() -> LoadedDeployment {
        LoadedDeployment {
            deployment_id: id::<DeploymentId>("chat-prod"),
            public_model: ModelAlias::new(ModelVariant::Qwen34BGguf.dir_name()).unwrap(),
            artifact_revision: id::<ArtifactRevision>("artifact-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
            task: TaskKind::Chat,
            backend: BackendKind::Cpu,
            precision: "mock".into(),
            execution_representation: "deterministic-text".into(),
            tokenizer_revision: None,
            readiness: ModelReadiness::Ready,
            capability: Capability {
                task: TaskKind::Chat,
                streaming: true,
                realtime: false,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                output_formats: BTreeSet::from([OutputFormat::Text]),
                max_input_bytes: 4096,
                max_context_tokens: Some(4096),
                max_output_tokens: Some(128),
            },
        }
    }

    fn register(
        registry: &WorkerRegistry,
        worker_name: &str,
        incarnation_name: &str,
        worker_client: WorkerClient,
        status_sequence: u64,
    ) {
        let loaded = deployment();
        let descriptor = WorkerDescriptor {
            schema_version: PROTOCOL_V1,
            supported_protocol_versions: vec![PROTOCOL_V1],
            worker_id: id::<WorkerId>(worker_name),
            node_id: id::<NodeId>("node-a"),
            incarnation_id: id(incarnation_name),
            build_version: "test".into(),
            assignment: DeviceAssignment::Cpu {
                thread_budget: 1,
                affinity: Vec::new(),
                host_memory_limit_bytes: 1024,
            },
            features: BTreeSet::from([WorkerFeature::Streaming, WorkerFeature::Cancellation]),
        };
        registry
            .approve(ApprovedWorker {
                descriptor: descriptor.clone(),
                client: worker_client,
                approved_deployments: BTreeMap::from([(
                    loaded.deployment_id.clone(),
                    ApprovedDeployment::from_loaded(&loaded),
                )]),
                validated_capacity: 1,
            })
            .unwrap();
        registry
            .observe_status(WorkerStatus {
                schema_version: PROTOCOL_V1,
                worker_id: descriptor.worker_id,
                node_id: descriptor.node_id,
                incarnation_id: descriptor.incarnation_id,
                status_sequence,
                process_state: WorkerProcessState::Running,
                deployments: vec![loaded],
                capacity: CapacitySnapshot {
                    max_active_invocations: 1,
                    active_invocations: 0,
                    max_queued_invocations: 0,
                    queued_invocations: 0,
                    max_sessions: 0,
                    reserved_sessions: 0,
                    available_admission_credits: 1,
                    outstanding_cost_units: 0,
                },
            })
            .unwrap();
    }

    fn dispatcher(registry: WorkerRegistry) -> RemoteChatDispatcher {
        RemoteChatDispatcher::new(
            registry,
            RemoteChatDispatchConfig {
                public_model_variant: ModelVariant::Qwen34BGguf,
                deployment_id: id::<DeploymentId>("chat-prod"),
                policy_revision: id::<PolicyRevision>("policy-v1"),
                backend_policy: BackendPolicy::ANY,
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
            },
        )
        .unwrap()
    }

    fn request() -> ChatExecutionRequest {
        ChatExecutionRequest {
            variant: ModelVariant::Qwen34BGguf,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".into(),
            }],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            presence_penalty: None,
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        }
    }

    #[test]
    fn local_reservation_balances_concurrent_dispatches() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client("http://127.0.0.1:19101"),
            1,
        );
        register(
            &registry,
            "worker-b",
            "inc-b",
            client("http://127.0.0.1:19102"),
            1,
        );
        let dispatcher = dispatcher(registry);

        let first = dispatcher.select(&request(), false).unwrap();
        assert_eq!(first.key.worker_id.as_str(), "worker-a");
        let second = dispatcher.select(&request(), false).unwrap();
        assert_eq!(second.key.worker_id.as_str(), "worker-b");

        drop(first);
        drop(second);
    }

    #[tokio::test]
    async fn authoritative_capacity_rejection_is_not_retried() {
        let mock_config = MockWorkerConfig {
            worker_id: id::<WorkerId>("worker-a"),
            node_id: id::<NodeId>("node-a"),
            incarnation_id: id("inc-a"),
            deployment_id: id::<DeploymentId>("chat-prod"),
            public_model: ModelAlias::new(ModelVariant::Qwen34BGguf.dir_name()).unwrap(),
            model_generation: ModelGeneration::new(1).unwrap(),
            credentials: credentials(),
            output_cadence: Duration::from_millis(5),
            fault: MockFault::Hang,
            ..MockWorkerConfig::default()
        };
        let worker = MockWorker::spawn(mock_config).await.unwrap();
        let worker_client = client(&worker.endpoint());
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(&registry, "worker-a", "inc-a", worker_client.clone(), 1);
        register(
            &registry,
            "worker-b",
            "inc-b",
            client("http://127.0.0.1:19102"),
            1,
        );

        // Occupy worker-a outside the registry so its last status is stale in
        // exactly the way a real multi-gateway deployment can observe.
        let pinned = RemoteChatExecution::new(
            worker_client,
            RemoteChatExecutionConfig {
                public_model_variant: ModelVariant::Qwen34BGguf,
                expected_worker_incarnation: id("inc-a"),
                deployment_id: id::<DeploymentId>("chat-prod"),
                expected_model_generation: ModelGeneration::new(1).unwrap(),
                policy_revision: id::<PolicyRevision>("policy-v1"),
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
            },
        )
        .unwrap();
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        let occupying_stream =
            spawn_remote_chat_stream_with_execution(&pinned, 2, &context, request())
                .await
                .unwrap();
        assert_eq!(worker.active_invocations(), 1);

        let error = dispatcher(registry)
            .generate(2, &context, request())
            .await
            .unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(error.message.contains("capacity"), "{}", error.message);

        drop(occupying_stream);
    }
}

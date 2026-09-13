//! Multi-worker routing for the migrated public chat slice.
//!
//! The registry narrows candidates using fresh, receiver-clock status, but the
//! chosen worker remains authoritative for admission. A dispatcher may make
//! one alternate attempt only when the first attempt is provably unaccepted;
//! uncertain acceptance and accepted execution never fail over.

use std::time::{Duration, Instant};

use izwi_core::{ChatGeneration, ModelVariant};
use izwi_serving_client::{DeadlinePhase, InvocationStream, WorkerClientError};
use izwi_serving_protocol::{
    CancellationBehavior, DeploymentId, InputFormat, ModelAlias, OutputFormat, PolicyRevision,
    RejectionCode, TaskKind, PROTOCOL_V1,
};
use tokio::sync::mpsc;

use super::chat::{
    collect_started_remote_chat, map_worker_client_error, prepare_remote_chat_invocation,
    retarget_remote_chat_invocation, spawn_started_remote_chat_stream_with_execution,
    start_remote_chat_invocation, ChatExecutionRequest, ChatStreamEvent, RemoteChatExecution,
    RemoteChatExecutionConfig,
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

    pub(crate) const fn max_output_tokens(&self) -> u32 {
        self.config.max_output_tokens
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
        let StartedDispatch {
            remote,
            mut stream,
            selected,
            started,
        } = self
            .start(request_timeout_secs, context, request, false)
            .await?;
        let key = selected.key.clone();
        let _dispatch = selected.dispatch;
        let registry = self.registry.clone();
        collect_started_remote_chat(&remote, &mut stream, started, move |error| {
            report_stream_error(&registry, &key, error);
        })
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
        let StartedDispatch {
            remote,
            stream,
            selected,
            ..
        } = self
            .start(request_timeout_secs, context, request, true)
            .await?;
        let registry = self.registry.clone();
        let key = selected.key.clone();
        let mut worker_events =
            spawn_started_remote_chat_stream_with_execution(&remote, stream, move |error| {
                report_stream_error(&registry, &key, error)
            });

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

    async fn start(
        &self,
        request_timeout_secs: u64,
        context: &RequestContext,
        request: ChatExecutionRequest,
        streaming: bool,
    ) -> Result<StartedDispatch, ApiError> {
        let selection = self.selection_request(&request, streaming)?;
        let mut selected = self
            .registry
            .select_and_reserve(&selection)
            .map_err(map_registry_error)?;
        let remote = self.execution_for(&selected)?;
        let invocation =
            prepare_remote_chat_invocation(&remote, request_timeout_secs, context, request)?;
        let started = Instant::now();

        match start_remote_chat_invocation(&remote, invocation.clone()).await {
            Ok(stream) => {
                selected
                    .dispatch
                    .mark_accepted()
                    .map_err(map_registry_error)?;
                return Ok(StartedDispatch {
                    remote,
                    stream,
                    selected,
                    started,
                });
            }
            Err(error) => {
                report_start_error(&self.registry, &selected, &error);
                if !retryable_before_acceptance(&error) {
                    return Err(map_worker_client_error(error));
                }
            }
        }

        let excluded = selected.key.clone();
        drop(selected);
        let mut alternate = self
            .registry
            .select_and_reserve_excluding(&selection, Some(&excluded))
            .map_err(map_registry_error)?;
        let alternate_remote = self.execution_for(&alternate)?;
        let remaining = context
            .remaining_budget(Duration::from_secs(request_timeout_secs.max(1)))
            .filter(|budget| !budget.is_zero())
            .ok_or_else(|| ApiError {
                status: axum::http::StatusCode::REQUEST_TIMEOUT,
                message: "Chat request deadline expired before alternate dispatch".into(),
            })?;
        let alternate_invocation =
            retarget_remote_chat_invocation(&invocation, &alternate_remote, remaining)?;
        match start_remote_chat_invocation(&alternate_remote, alternate_invocation).await {
            Ok(stream) => {
                alternate
                    .dispatch
                    .mark_accepted()
                    .map_err(map_registry_error)?;
                Ok(StartedDispatch {
                    remote: alternate_remote,
                    stream,
                    selected: alternate,
                    started,
                })
            }
            Err(error) => {
                report_start_error(&self.registry, &alternate, &error);
                Err(map_worker_client_error(error))
            }
        }
    }

    fn selection_request(
        &self,
        request: &ChatExecutionRequest,
        streaming: bool,
    ) -> Result<WorkerSelectionRequest, ApiError> {
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
        Ok(WorkerSelectionRequest {
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
        })
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

struct StartedDispatch {
    remote: RemoteChatExecution,
    stream: InvocationStream,
    selected: SelectedWorker,
    started: Instant,
}

fn retryable_before_acceptance(error: &WorkerClientError) -> bool {
    match error {
        WorkerClientError::ConnectionNotEstablished(_)
        | WorkerClientError::Deadline(DeadlinePhase::InFlightPermit) => true,
        WorkerClientError::Rejected { rejection } if !rejection.accepted => matches!(
            rejection.code,
            RejectionCode::CapacityExhausted
                | RejectionCode::QueueWaitExceeded
                | RejectionCode::WrongWorkerIncarnation
                | RejectionCode::WrongModelGeneration
                | RejectionCode::UnknownDeployment
                | RejectionCode::ModelNotReady
                | RejectionCode::WorkerDraining
        ),
        _ => false,
    }
}

fn report_start_error(
    registry: &WorkerRegistry,
    selected: &SelectedWorker,
    error: &WorkerClientError,
) {
    if matches!(error, WorkerClientError::Rejected { rejection } if !rejection.accepted) {
        let _ = registry.report_worker_reachable(&selected.key);
    } else {
        report_stream_error(registry, &selected.key, error);
    }
}

fn report_stream_error(
    registry: &WorkerRegistry,
    key: &crate::worker_registry::WorkerInstanceKey,
    error: &WorkerClientError,
) {
    if counts_as_transport_failure(error) {
        let _ = registry.report_worker_transport_failure(key);
    }
}

fn counts_as_transport_failure(error: &WorkerClientError) -> bool {
    matches!(
        error,
        WorkerClientError::ConnectionNotEstablished(_)
            | WorkerClientError::Transport(_)
            | WorkerClientError::HttpStatus { .. }
            | WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders)
            | WorkerClientError::Deadline(DeadlinePhase::StreamProgress)
            | WorkerClientError::Deadline(DeadlinePhase::TotalInvocation)
            | WorkerClientError::ResponseTooLarge { .. }
            | WorkerClientError::InvalidJson(_)
            | WorkerClientError::Ndjson(_)
            | WorkerClientError::Protocol(_)
            | WorkerClientError::InterruptedUnknown
    )
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
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    use axum::body::{Body, Bytes};
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::routing::post;
    use axum::Router;
    use izwi_core::{ChatMessage, ChatRequestConfig, ChatRole};
    use izwi_hooks::Principal;
    use izwi_serving_client::{WorkerClient, WorkerClientConfig};
    use izwi_serving_protocol::{
        ArtifactRevision, BackendKind, Capability, CapacitySnapshot, CredentialId,
        DeviceAssignment, FinishReason, InvocationEvent, InvocationEventKind, InvocationRejection,
        InvocationRequest, LoadedDeployment, ModelGeneration, ModelReadiness, NodeId,
        ServiceBearerToken, ServiceCredentials, Usage, WorkerDescriptor, WorkerFeature, WorkerId,
        WorkerProcessState, WorkerStatus, INVOCATIONS_PATH, NDJSON_MEDIA_TYPE,
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

    #[derive(Clone)]
    enum ScriptedResponse {
        Reject(RejectionCode),
        AcceptedWithoutAcknowledgement,
        PartialThenDisconnect,
        Success(String),
    }

    #[derive(Clone)]
    struct ScriptedState {
        response: ScriptedResponse,
        delay: Duration,
        requests: Arc<Mutex<Vec<InvocationRequest>>>,
    }

    struct ScriptedWorker {
        address: SocketAddr,
        requests: Arc<Mutex<Vec<InvocationRequest>>>,
        server: tokio::task::JoinHandle<()>,
    }

    impl ScriptedWorker {
        async fn spawn(response: ScriptedResponse, delay: Duration) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let state = ScriptedState {
                response,
                delay,
                requests: Arc::clone(&requests),
            };
            let app = Router::new()
                .route(INVOCATIONS_PATH, post(scripted_invoke))
                .with_state(state);
            let server = tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            Self {
                address,
                requests,
                server,
            }
        }

        fn endpoint(&self) -> String {
            format!("http://{}/", self.address)
        }

        fn requests(&self) -> Vec<InvocationRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for ScriptedWorker {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    async fn scripted_invoke(State(state): State<ScriptedState>, body: Bytes) -> Response<Body> {
        let request: InvocationRequest = serde_json::from_slice(&body).unwrap();
        state.requests.lock().unwrap().push(request.clone());
        if !state.delay.is_zero() {
            tokio::time::sleep(state.delay).await;
        }
        match state.response {
            ScriptedResponse::Reject(code) => Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&InvocationRejection::new(
                        request.request_id,
                        request.attempt_id,
                        code,
                        "scripted rejection",
                    ))
                    .unwrap(),
                ))
                .unwrap(),
            ScriptedResponse::AcceptedWithoutAcknowledgement => Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
                .body(Body::empty())
                .unwrap(),
            ScriptedResponse::PartialThenDisconnect => accepted_response(
                &request,
                vec![InvocationEventKind::TextDelta {
                    text: "partial".into(),
                }],
            ),
            ScriptedResponse::Success(text) => accepted_response(
                &request,
                vec![
                    InvocationEventKind::TextDelta { text },
                    InvocationEventKind::Completed {
                        finish_reason: FinishReason::Stop,
                        usage: Some(Usage {
                            input_tokens: 1,
                            output_tokens: 1,
                        }),
                    },
                ],
            ),
        }
    }

    fn accepted_response(
        request: &InvocationRequest,
        following: Vec<InvocationEventKind>,
    ) -> Response<Body> {
        let mut events = vec![InvocationEventKind::Accepted {
            worker_id: id::<WorkerId>("scripted-worker"),
            node_id: id::<NodeId>("node-a"),
            incarnation_id: request.expected_worker_incarnation.clone(),
            deployment_id: request.deployment_id.clone(),
            model_generation: request.expected_model_generation,
        }];
        events.extend(following);
        let mut encoded = Vec::new();
        for (sequence, event) in events.into_iter().enumerate() {
            encoded.extend(
                serde_json::to_vec(&InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request.request_id.clone(),
                    attempt_id: request.attempt_id.clone(),
                    sequence: sequence as u64,
                    event,
                })
                .unwrap(),
            );
            encoded.push(b'\n');
        }
        Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
            .body(Body::from(encoded))
            .unwrap()
    }

    async fn refused_endpoint() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        format!("http://{address}/")
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
        let selection = dispatcher.selection_request(&request(), false).unwrap();

        let first = dispatcher.registry.select_and_reserve(&selection).unwrap();
        assert_eq!(first.key.worker_id.as_str(), "worker-a");
        let second = dispatcher.registry.select_and_reserve(&selection).unwrap();
        assert_eq!(second.key.worker_id.as_str(), "worker-b");

        drop(first);
        drop(second);
    }

    #[test]
    fn retry_classifier_is_an_explicit_allowlist() {
        let rejection = |code| WorkerClientError::Rejected {
            rejection: InvocationRejection::new(id("request-1"), id("attempt-1"), code, "test"),
        };
        for code in [
            RejectionCode::CapacityExhausted,
            RejectionCode::QueueWaitExceeded,
            RejectionCode::WrongWorkerIncarnation,
            RejectionCode::WrongModelGeneration,
            RejectionCode::UnknownDeployment,
            RejectionCode::ModelNotReady,
            RejectionCode::WorkerDraining,
        ] {
            assert!(retryable_before_acceptance(&rejection(code)), "{code:?}");
        }
        for code in [
            RejectionCode::Unauthenticated,
            RejectionCode::Unauthorized,
            RejectionCode::UnsupportedProtocolVersion,
            RejectionCode::IncompatibleTask,
            RejectionCode::InvalidRequest,
            RejectionCode::DuplicateAttemptConflict,
            RejectionCode::PolicyDenied,
        ] {
            assert!(!retryable_before_acceptance(&rejection(code)), "{code:?}");
        }
        assert!(retryable_before_acceptance(&WorkerClientError::Deadline(
            DeadlinePhase::InFlightPermit,
        )));
        assert!(!retryable_before_acceptance(&WorkerClientError::Deadline(
            DeadlinePhase::ResponseHeaders,
        )));
        assert!(!retryable_before_acceptance(
            &WorkerClientError::HttpStatus {
                status: "503".parse().unwrap(),
                body: "generic".into(),
            },
        ));

        let mut invalid_accepted = match rejection(RejectionCode::CapacityExhausted) {
            WorkerClientError::Rejected { rejection } => rejection,
            _ => unreachable!(),
        };
        invalid_accepted.accepted = true;
        assert!(!retryable_before_acceptance(&WorkerClientError::Rejected {
            rejection: invalid_accepted,
        },));
    }

    #[tokio::test]
    async fn connection_not_established_retries_once_for_streaming() {
        let worker_b =
            ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&refused_endpoint().await),
            1,
        );
        register(
            &registry,
            "worker-b",
            "inc-b",
            client(&worker_b.endpoint()),
            1,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

        let mut events = dispatcher(registry)
            .stream(2, &context, request())
            .await
            .unwrap();
        assert!(matches!(
            events.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        assert!(matches!(
            events.recv().await,
            Some(ChatStreamEvent::Delta(ref text)) if text == "from-b"
        ));
        assert!(matches!(
            events.recv().await,
            Some(ChatStreamEvent::Completed(_))
        ));
        assert_eq!(worker_b.requests().len(), 1);
    }

    #[tokio::test]
    async fn capacity_rejection_retries_with_same_request_and_fresh_attempt() {
        let worker_a = ScriptedWorker::spawn(
            ScriptedResponse::Reject(RejectionCode::CapacityExhausted),
            Duration::from_millis(25),
        )
        .await;
        let worker_b =
            ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker_a.endpoint()),
            1,
        );
        register(
            &registry,
            "worker-b",
            "inc-b",
            client(&worker_b.endpoint()),
            1,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

        let generation = dispatcher(registry)
            .generate(2, &context, request())
            .await
            .unwrap();
        assert_eq!(generation.text, "from-b");
        let first = worker_a.requests();
        let alternate = worker_b.requests();
        assert_eq!(first.len(), 1);
        assert_eq!(alternate.len(), 1);
        assert_eq!(first[0].request_id, alternate[0].request_id);
        assert_ne!(first[0].attempt_id, alternate[0].attempt_id);
        assert!(alternate[0].remaining_time_ms < first[0].remaining_time_ms);
    }

    #[tokio::test]
    async fn uncertain_acceptance_and_partial_output_never_retry() {
        for first_response in [
            ScriptedResponse::AcceptedWithoutAcknowledgement,
            ScriptedResponse::PartialThenDisconnect,
        ] {
            let worker_a = ScriptedWorker::spawn(first_response, Duration::ZERO).await;
            let worker_b = ScriptedWorker::spawn(
                ScriptedResponse::Success("must-not-run".into()),
                Duration::ZERO,
            )
            .await;
            let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
            register(
                &registry,
                "worker-a",
                "inc-a",
                client(&worker_a.endpoint()),
                1,
            );
            register(
                &registry,
                "worker-b",
                "inc-b",
                client(&worker_b.endpoint()),
                1,
            );
            let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

            let error = dispatcher(registry)
                .generate(2, &context, request())
                .await
                .unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_GATEWAY);
            assert!(worker_b.requests().is_empty());
        }
    }

    #[tokio::test]
    async fn retry_is_bounded_to_one_alternate() {
        let worker_c = ScriptedWorker::spawn(
            ScriptedResponse::Success("must-not-run".into()),
            Duration::ZERO,
        )
        .await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&refused_endpoint().await),
            1,
        );
        register(
            &registry,
            "worker-b",
            "inc-b",
            client(&refused_endpoint().await),
            1,
        );
        register(
            &registry,
            "worker-c",
            "inc-c",
            client(&worker_c.endpoint()),
            1,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

        let error = dispatcher(registry)
            .generate(2, &context, request())
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert!(worker_c.requests().is_empty());
    }
}

//! Multi-worker routing for the migrated public chat slice.
//!
//! The registry narrows candidates using fresh, receiver-clock status, but the
//! chosen worker remains authoritative for admission. A dispatcher may make
//! one alternate attempt only when the first attempt is provably unaccepted;
//! uncertain acceptance and accepted execution never fail over.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use izwi_core::{ChatGeneration, ModelVariant};
use izwi_serving_client::{DeadlinePhase, InvocationStream, WorkerClientError};
use izwi_serving_protocol::{
    CancellationBehavior, DeploymentId, InputFormat, ModelAlias, OutputFormat, PolicyRevision,
    TaskKind, PROTOCOL_V1,
};
use tokio::sync::mpsc;

use super::chat::{
    collect_started_remote_chat, map_worker_client_error, prepare_remote_chat_invocation,
    retarget_remote_chat_invocation, spawn_started_remote_chat_stream,
    start_remote_chat_invocation_with_tenant, ChatExecutionRequest, ChatStreamEvent,
    RemoteChatExecution, RemoteChatExecutionConfig, RemoteWorkerContract,
};
use super::fleet_coordinator::{FleetClaimGuard, FleetCoordinator};
use super::prefix_router::{PrefixAffinityConfig, PrefixRouter, PromptBlocks, RequestPrefix};
use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::gateway_tenant_concurrency::{BoundTenantWorkLease, UnboundTenantWorkLease};
use crate::worker_registry::{
    BackendPolicy, FleetCapacityView, NoFleetCapacity, PrefixAffinity, SelectedWorker,
    WorkerInstanceKey, WorkerRegistry, WorkerRegistryError, WorkerSelectionRequest,
};

const RETRY_BACKOFF_BASE_MS: u64 = 10;
const RETRY_BACKOFF_JITTER_MS: u64 = 10;
const RETRY_BACKOFF_MAX_MS: u64 = 100;
#[derive(Debug, Clone)]
pub struct RemoteChatDispatchConfig {
    pub public_model_variant: ModelVariant,
    pub deployment_id: DeploymentId,
    pub policy_revision: PolicyRevision,
    pub backend_policy: BackendPolicy,
    pub max_queue_wait: Duration,
    pub max_output_tokens: u32,
    pub max_output_bytes: u64,
    pub slow_consumer_timeout: Duration,
    /// Prefix-aware routing: prefer the worker that was recently routed the
    /// longest leading part of this prompt. `None` routes on load alone.
    pub prefix_affinity: Option<PrefixAffinityConfig>,
}

impl RemoteChatDispatchConfig {
    fn validate(&self) -> Result<(), ApiError> {
        if self.max_output_tokens == 0
            || self.max_output_bytes == 0
            || self.slow_consumer_timeout.is_zero()
        {
            return Err(ApiError::internal(
                "Remote chat dispatch output limits and slow-consumer timeout must be non-zero",
            ));
        }
        if let Some(prefix_affinity) = self.prefix_affinity.as_ref() {
            prefix_affinity.validate()?;
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
    fleet: Option<Arc<FleetCoordinator>>,
    prefix: Option<Arc<PrefixRouter>>,
    /// The gateway's shutdown broadcast to live streams.
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
}

impl RemoteChatDispatcher {
    pub fn new(
        registry: WorkerRegistry,
        config: RemoteChatDispatchConfig,
    ) -> Result<Self, ApiError> {
        config.validate()?;
        let prefix = config
            .prefix_affinity
            .map(|prefix_affinity| Arc::new(PrefixRouter::new(prefix_affinity)));
        Ok(Self {
            registry,
            config,
            fleet: None,
            prefix,
            shutdown: None,
        })
    }

    /// Attach cluster capacity coordination. Without a coordinator the
    /// dispatcher behaves exactly as before (single-gateway profile).
    pub fn with_fleet_coordinator(mut self, coordinator: Arc<FleetCoordinator>) -> Self {
        self.fleet = Some(coordinator);
        self
    }

    /// End live streams with a terminal "shutting down" event once
    /// `shutdown` turns `true`.
    pub(crate) fn with_shutdown(mut self, shutdown: tokio::sync::watch::Receiver<bool>) -> Self {
        self.shutdown = Some(shutdown);
        self
    }

    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    pub(crate) const fn max_output_tokens(&self) -> u32 {
        self.config.max_output_tokens
    }

    /// The model this deployment serves under its public name.
    pub(crate) const fn public_model(&self) -> ModelVariant {
        self.config.public_model_variant
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
                    required_features: BTreeSet::new(),
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

    pub(crate) async fn generate(
        &self,
        request_timeout_secs: u64,
        context: &RequestContext,
        request: ChatExecutionRequest,
        tenant_work: UnboundTenantWorkLease,
    ) -> Result<ChatGeneration, ApiError> {
        let StartedDispatch {
            remote,
            mut stream,
            tenant_work,
            selected,
            started,
        } = self
            .start(request_timeout_secs, context, request, false, tenant_work)
            .await?;
        let key = selected.key.clone();
        let _dispatch = selected.dispatch;
        let registry = self.registry.clone();
        collect_started_remote_chat(
            &remote,
            &mut stream,
            started,
            move |error| {
                report_stream_error(&registry, &key, error);
            },
            tenant_work,
        )
        .await
    }

    /// Returns only after the selected worker has emitted a contract-valid
    /// accepted event. The forwarding task owns the local dispatch reservation
    /// through terminal delivery or public-consumer disconnect.
    pub(crate) async fn stream(
        &self,
        request_timeout_secs: u64,
        context: &RequestContext,
        request: ChatExecutionRequest,
        tenant_work: UnboundTenantWorkLease,
    ) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError> {
        let StartedDispatch {
            remote,
            stream,
            tenant_work,
            selected,
            ..
        } = self
            .start(request_timeout_secs, context, request, true, tenant_work)
            .await?;
        let registry = self.registry.clone();
        let key = selected.key.clone();
        // The relay task holds the registry reservation; dropping the
        // receiver propagates cancellation toward the exact accepted
        // attempt. There is intentionally no reselection.
        Ok(spawn_started_remote_chat_stream(
            &remote,
            stream,
            move |error| report_stream_error(&registry, &key, error),
            tenant_work,
            selected.dispatch,
            self.shutdown.clone(),
        ))
    }

    async fn start(
        &self,
        request_timeout_secs: u64,
        context: &RequestContext,
        request: ChatExecutionRequest,
        streaming: bool,
        tenant_work: UnboundTenantWorkLease,
    ) -> Result<StartedDispatch, ApiError> {
        let selection = self.selection_request(&request, streaming)?;
        // Prefix-aware routing: prefer a worker that recently served this
        // prompt's leading blocks, within the registry's load bound.
        let prefix = self
            .prefix
            .as_deref()
            .map(|router| (router, router.blocks(&request.messages)));
        let mut selected = self
            .select(&selection, prefix.as_ref(), None)
            .map_err(|error| self.selection_error(error, &selection, &request))?;
        // Best-effort cluster visibility: publish this dispatch to peer
        // gateways when coordinated, but never gate on it. A lost race only
        // costs one worker-arbitrated attempt through the existing alternate
        // path below. The claim covers only the transit to acceptance: from
        // then on the worker's own status reports the work.
        let fleet_claim = self.claim_for(&selected).await;
        let remote = self.execution_for(&selected)?;
        let invocation =
            prepare_remote_chat_invocation(&remote, request_timeout_secs, context, request)?;
        let started = Instant::now();

        let retry_delay = match start_remote_chat_invocation_with_tenant(
            &remote,
            invocation.clone(),
            tenant_work,
            &context.trace,
        )
        .await
        {
            Ok(started_invocation) => {
                selected
                    .dispatch
                    .mark_accepted()
                    .map_err(map_registry_error)?;
                drop(fleet_claim);
                if let Some((router, blocks)) = prefix.as_ref() {
                    router.record(&selected.key, blocks);
                }
                return Ok(StartedDispatch {
                    remote,
                    stream: started_invocation.stream,
                    tenant_work: started_invocation.tenant_work,
                    selected,
                    started,
                });
            }
            Err(failure) => {
                report_start_error(&self.registry, &selected, &failure.error);
                let Some(retry_tenant_work) = failure.retry_tenant_work else {
                    return Err(map_worker_client_error(failure.error));
                };
                if !retryable_before_acceptance(&failure.error) {
                    return Err(map_worker_client_error(failure.error));
                }
                (
                    retry_delay(&failure.error, invocation.request_id.as_str()),
                    retry_tenant_work,
                )
            }
        };

        let excluded = selected.key.clone();
        drop(selected);
        drop(fleet_claim);
        let remaining_before_backoff = context
            .remaining_budget(Duration::from_secs(request_timeout_secs.max(1)))
            .filter(|budget| !budget.is_zero())
            .ok_or_else(alternate_deadline_error)?;
        if retry_delay.0 >= remaining_before_backoff {
            return Err(alternate_deadline_error());
        }
        tokio::time::sleep(retry_delay.0).await;
        let mut alternate = self
            .select(&selection, prefix.as_ref(), Some(&excluded))
            .map_err(map_registry_error)?;
        let alternate_fleet_claim = self.claim_for(&alternate).await;
        let alternate_remote = self.execution_for(&alternate)?;
        let remaining = context
            .remaining_budget(Duration::from_secs(request_timeout_secs.max(1)))
            .filter(|budget| !budget.is_zero())
            .ok_or_else(alternate_deadline_error)?;
        let alternate_invocation =
            retarget_remote_chat_invocation(&invocation, &alternate_remote, remaining)?;
        match start_remote_chat_invocation_with_tenant(
            &alternate_remote,
            alternate_invocation,
            retry_delay.1,
            &context.trace,
        )
        .await
        {
            Ok(started_invocation) => {
                alternate
                    .dispatch
                    .mark_accepted()
                    .map_err(map_registry_error)?;
                drop(alternate_fleet_claim);
                if let Some((router, blocks)) = prefix.as_ref() {
                    router.record(&alternate.key, blocks);
                }
                Ok(StartedDispatch {
                    remote: alternate_remote,
                    stream: started_invocation.stream,
                    tenant_work: started_invocation.tenant_work,
                    selected: alternate,
                    started,
                })
            }
            Err(failure) => {
                report_start_error(&self.registry, &alternate, &failure.error);
                Err(map_worker_client_error(failure.error))
            }
        }
    }

    fn select(
        &self,
        selection: &WorkerSelectionRequest,
        prefix: Option<&(&PrefixRouter, PromptBlocks)>,
        excluded: Option<&WorkerInstanceKey>,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        let affinity = prefix.map(|(router, blocks)| RequestPrefix { router, blocks });
        let fleet = self.fleet.as_ref().map(|fleet| fleet.snapshot_view());
        let fleet: &dyn FleetCapacityView = match fleet.as_ref() {
            Some(fleet) => fleet,
            None => &NoFleetCapacity,
        };
        self.registry.select_and_reserve_routed(
            selection,
            excluded,
            affinity
                .as_ref()
                .map(|affinity| affinity as &dyn PrefixAffinity),
            fleet,
            Instant::now(),
        )
    }

    /// Publish one cluster capacity claim for a selected worker. Returns None
    /// without a coordinator, or when peers already hold every observable
    /// credit; both cases proceed to invoke and let the worker arbitrate.
    async fn claim_for(&self, selected: &SelectedWorker) -> Option<FleetClaimGuard> {
        let fleet = self.fleet.as_ref()?;
        fleet.claim(&selected.key, selected.available_credits).await
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
            required_features: request.required_worker_features(),
        })
    }

    /// No eligible worker is a 503, unless workers are ready and only the
    /// features the request's controls need are missing: that is the
    /// request's 400, as on a worker that refuses the control.
    fn selection_error(
        &self,
        error: WorkerRegistryError,
        selection: &WorkerSelectionRequest,
        request: &ChatExecutionRequest,
    ) -> ApiError {
        if error == WorkerRegistryError::NoEligibleWorker && !selection.required_features.is_empty()
        {
            let without_features = WorkerSelectionRequest {
                required_features: BTreeSet::new(),
                ..selection.clone()
            };
            if let (true, Some(parameter)) = (
                self.registry.has_fresh_compatible_worker(&without_features),
                request.first_feature_gated_parameter(),
            ) {
                return ApiError::unsupported_param(
                    parameter,
                    format!("No ready worker for this model supports `{parameter}`"),
                );
            }
        }
        map_registry_error(error)
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
                slow_consumer_timeout: self.config.slow_consumer_timeout,
            },
            worker_contract(selected),
        )
    }
}

fn worker_contract(selected: &SelectedWorker) -> RemoteWorkerContract {
    RemoteWorkerContract {
        protocol_version: selected.protocol_version,
        features: selected.features.clone(),
    }
}

fn alternate_deadline_error() -> ApiError {
    ApiError {
        status: axum::http::StatusCode::REQUEST_TIMEOUT,
        message: "Chat request deadline expired before alternate dispatch".into(),
        detail: None,
    }
}

fn retry_delay(error: &WorkerClientError, request_id: &str) -> Duration {
    let advised_ms = match error {
        WorkerClientError::Rejected { rejection } => rejection.retry_after_ms,
        _ => None,
    }
    .unwrap_or(RETRY_BACKOFF_BASE_MS)
    .clamp(RETRY_BACKOFF_BASE_MS, RETRY_BACKOFF_MAX_MS);
    let jitter_ceiling = RETRY_BACKOFF_MAX_MS.saturating_sub(advised_ms);
    let jitter_window = RETRY_BACKOFF_JITTER_MS.min(jitter_ceiling);
    let jitter_ms = stable_request_jitter(request_id, jitter_window);
    Duration::from_millis(advised_ms.saturating_add(jitter_ms))
}

fn stable_request_jitter(request_id: &str, inclusive_max_ms: u64) -> u64 {
    if inclusive_max_ms == 0 {
        return 0;
    }
    let hash = request_id.bytes().fold(2_166_136_261u64, |state, byte| {
        state.wrapping_mul(16_777_619) ^ u64::from(byte)
    });
    hash % inclusive_max_ms.saturating_add(1)
}

struct StartedDispatch {
    remote: RemoteChatExecution,
    stream: InvocationStream,
    tenant_work: BoundTenantWorkLease,
    selected: SelectedWorker,
    started: Instant,
}

fn retryable_before_acceptance(error: &WorkerClientError) -> bool {
    match error {
        WorkerClientError::ConnectionNotEstablished(_)
        | WorkerClientError::Deadline(DeadlinePhase::InFlightPermit) => true,
        WorkerClientError::Rejected { rejection } if !rejection.accepted => {
            crate::error::rejection_class(rejection.code).retryable()
        }
        _ => false,
    }
}

fn report_start_error(
    registry: &WorkerRegistry,
    selected: &SelectedWorker,
    error: &WorkerClientError,
) {
    // A typed rejection or an unknown admission outcome is an authenticated
    // answer from a live worker (busy at worst), never a transport strike.
    if matches!(error, WorkerClientError::Rejected { rejection } if !rejection.accepted)
        || matches!(error, WorkerClientError::AdmissionUnknown { .. })
    {
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
            | WorkerClientError::Deadline(DeadlinePhase::InvocationAdmission)
            | WorkerClientError::Deadline(DeadlinePhase::FirstOutput)
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
    use izwi_serving_protocol::RejectionCode;
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
        ArtifactRevision, BackendKind, Capability, CapacitySnapshot, ChatContextPolicy,
        ChatParameters as WorkerChatParameters, ChatResponseFormat,
        ChatTruncation as WorkerChatTruncation, CredentialId, DeviceAssignment, FinishReason,
        IncarnationId, InvocationAdmissionUnknown, InvocationEvent, InvocationEventKind,
        InvocationInput, InvocationRejection, InvocationRequest, LoadedDeployment, ModelGeneration,
        ModelReadiness, NodeId, SchemaVersion, ServiceBearerToken, ServiceCredentials, Usage,
        WorkerDescriptor, WorkerFeature, WorkerId, WorkerProcessState, WorkerStatus,
        INVOCATIONS_PATH, NDJSON_MEDIA_TYPE,
    };

    use crate::gateway_fleet::FleetStore;
    use crate::db::StoreDatabase;

    use crate::worker_registry::{
        ApprovedDeployment, ApprovedWorker, RoutingConfig, WorkerRegistryConfig,
    };

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
        AdmissionUnknown,
        AcceptedWithoutAcknowledgement,
        PartialThenDisconnect,
        ManyDeltas(usize),
        Success(String),
        /// Accepts, then keeps the stream open without further events: an
        /// in-flight invocation that holds one worker credit.
        AcceptThenHold,
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
            ScriptedResponse::AdmissionUnknown => Response::builder()
                .status(StatusCode::GATEWAY_TIMEOUT)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&InvocationAdmissionUnknown {
                        schema_version: PROTOCOL_V1,
                        request_id: request.request_id,
                        attempt_id: request.attempt_id,
                        message: "scripted admission timeout".into(),
                    })
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
                    logprobs: None,
                }],
            ),
            ScriptedResponse::ManyDeltas(count) => {
                let mut following = Vec::with_capacity(count.saturating_add(1));
                following.extend((0..count).map(|_| InvocationEventKind::TextDelta {
                    text: "x".into(),
                    logprobs: None,
                }));
                following.push(InvocationEventKind::Completed {
                    finish_reason: FinishReason::Stop,
                    usage: Some(Usage {
                        input_tokens: 1,
                        output_tokens: u64::try_from(count).unwrap_or(u64::MAX),
                        cached_tokens: None,
                    }),
                });
                accepted_response(&request, following)
            }
            ScriptedResponse::AcceptThenHold => {
                use futures::StreamExt as _;

                let accepted = accepted_response(&request, Vec::new());
                let first = axum::body::to_bytes(accepted.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let held = futures::stream::once(async move { Ok::<_, std::io::Error>(first) })
                    .chain(futures::stream::pending());
                Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
                    .body(Body::from_stream(held))
                    .unwrap()
            }
            ScriptedResponse::Success(text) => accepted_response(
                &request,
                vec![
                    InvocationEventKind::TextDelta {
                        text,
                        logprobs: None,
                    },
                    InvocationEventKind::Completed {
                        finish_reason: FinishReason::Stop,
                        usage: Some(Usage {
                            input_tokens: 1,
                            output_tokens: 1,
                            cached_tokens: None,
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
            public_model: ModelAlias::new(ModelVariant::Qwen354BGguf.dir_name()).unwrap(),
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
            kv_cache_usage_pct: None,
            prefix_hits_total: None,
            prefix_queries_total: None,
            prefix_evictions_total: None,
            kv_host_pages: None,
            kv_demotions_total: None,
            kv_promotions_total: None,
            kv_promotion_latency_avg_seconds: None,
            tokens_out_per_s_ema: None,
            observation_cost_units: None,
        }
    }

    fn register(
        registry: &WorkerRegistry,
        worker_name: &str,
        incarnation_name: &str,
        worker_client: WorkerClient,
        status_sequence: u64,
    ) {
        register_with_credits(
            registry,
            worker_name,
            incarnation_name,
            worker_client,
            status_sequence,
            1,
        );
    }

    fn register_with_credits(
        registry: &WorkerRegistry,
        worker_name: &str,
        incarnation_name: &str,
        worker_client: WorkerClient,
        status_sequence: u64,
        credits: u32,
    ) {
        register_with_surface(
            registry,
            worker_name,
            incarnation_name,
            worker_client,
            status_sequence,
            credits,
            PROTOCOL_V1,
            izwi_serving_worker::test_support::minor4_chat_features(),
        );
    }

    /// A worker before minor 4: it advertises its own minor and no chat
    /// control feature.
    fn register_minor3(
        registry: &WorkerRegistry,
        worker_name: &str,
        incarnation_name: &str,
        worker_client: WorkerClient,
    ) {
        register_with_surface(
            registry,
            worker_name,
            incarnation_name,
            worker_client,
            1,
            1,
            SchemaVersion::new(1, 3),
            BTreeSet::new(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn register_with_surface(
        registry: &WorkerRegistry,
        worker_name: &str,
        incarnation_name: &str,
        worker_client: WorkerClient,
        status_sequence: u64,
        credits: u32,
        protocol_version: SchemaVersion,
        chat_features: BTreeSet<WorkerFeature>,
    ) {
        let loaded = deployment();
        let mut features = BTreeSet::from([WorkerFeature::Streaming, WorkerFeature::Cancellation]);
        features.extend(chat_features);
        let descriptor = WorkerDescriptor {
            schema_version: protocol_version,
            supported_protocol_versions: vec![protocol_version],
            worker_id: id::<WorkerId>(worker_name),
            node_id: id::<NodeId>("node-a"),
            incarnation_id: id(incarnation_name),
            build_version: "test".into(),
            assignment: DeviceAssignment::Cpu {
                thread_budget: 1,
                affinity: Vec::new(),
                host_memory_limit_bytes: 1024,
            },
            features,
        };
        registry
            .approve(ApprovedWorker {
                descriptor: descriptor.clone(),
                client: worker_client,
                approved_deployments: BTreeMap::from([(
                    loaded.deployment_id.clone(),
                    ApprovedDeployment::from_loaded(&loaded),
                )]),
                validated_capacity: credits,
            })
            .unwrap();
        registry
            .observe_status(worker_status(
                worker_name,
                incarnation_name,
                status_sequence,
                credits,
                0,
            ))
            .unwrap();
    }

    fn worker_status(
        worker_name: &str,
        incarnation_name: &str,
        status_sequence: u64,
        credits: u32,
        active: u32,
    ) -> WorkerStatus {
        WorkerStatus {
            schema_version: PROTOCOL_V1,
            worker_id: id::<WorkerId>(worker_name),
            node_id: id::<NodeId>("node-a"),
            incarnation_id: id(incarnation_name),
            status_sequence,
            process_state: WorkerProcessState::Running,
            deployments: vec![deployment()],
            capacity: CapacitySnapshot {
                max_active_invocations: credits,
                active_invocations: active,
                max_queued_invocations: 0,
                queued_invocations: 0,
                max_sessions: 0,
                reserved_sessions: 0,
                available_admission_credits: credits.saturating_sub(active),
                outstanding_cost_units: 0,
            },
        }
    }

    fn dispatcher(registry: WorkerRegistry) -> RemoteChatDispatcher {
        dispatcher_with_prefix(registry, None)
    }

    fn dispatcher_with_prefix(
        registry: WorkerRegistry,
        prefix_affinity: Option<PrefixAffinityConfig>,
    ) -> RemoteChatDispatcher {
        RemoteChatDispatcher::new(
            registry,
            RemoteChatDispatchConfig {
                public_model_variant: ModelVariant::Qwen354BGguf,
                deployment_id: id::<DeploymentId>("chat-prod"),
                policy_revision: id::<PolicyRevision>("policy-v1"),
                backend_policy: BackendPolicy::ANY,
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
                prefix_affinity,
            },
        )
        .unwrap()
    }

    fn request() -> ChatExecutionRequest {
        ChatExecutionRequest {
            variant: ModelVariant::Qwen354BGguf,
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
            seed: None,
            stop: Vec::new(),
            logprobs: None,
            top_logprobs: None,
            response_format_json_object: false,
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        }
    }

    fn tenant_work() -> UnboundTenantWorkLease {
        crate::gateway_tenant_concurrency::GatewayTenantConcurrency::new(
            crate::gateway_tenant_concurrency::GatewayTenantConcurrencyConfig::new(1, 1).unwrap(),
        )
        .try_reserve([7; 32])
        .unwrap()
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

    #[test]
    fn retry_backoff_honors_advice_with_bounded_stable_jitter() {
        let rejection = |retry_after_ms| {
            let mut rejection = InvocationRejection::new(
                id("request-1"),
                id("attempt-1"),
                RejectionCode::CapacityExhausted,
                "test",
            );
            rejection.retry_after_ms = retry_after_ms;
            WorkerClientError::Rejected { rejection }
        };

        let default_delay = retry_delay(&rejection(None), "request-1");
        assert!(default_delay >= Duration::from_millis(RETRY_BACKOFF_BASE_MS));
        assert!(
            default_delay <= Duration::from_millis(RETRY_BACKOFF_BASE_MS + RETRY_BACKOFF_JITTER_MS)
        );
        assert_eq!(
            default_delay,
            retry_delay(&rejection(None), "request-1"),
            "one request must use a stable retry delay"
        );

        let advised = retry_delay(&rejection(Some(95)), "request-1");
        assert!(advised >= Duration::from_millis(95));
        assert!(advised <= Duration::from_millis(RETRY_BACKOFF_MAX_MS));
        assert_eq!(
            retry_delay(&rejection(Some(u64::MAX)), "request-1"),
            Duration::from_millis(RETRY_BACKOFF_MAX_MS),
            "untrusted worker advice must be clamped"
        );
    }

    #[tokio::test]
    async fn worker_receives_the_resolved_sampling_profile() {
        let worker =
            ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

        // Qwen3.5-4B thinks by default; its published profile is T 1.0 and
        // top-p 0.95, never the worker's greedy default. Stop strings travel
        // with it (the worker applies them in the engine).
        let mut request = request();
        request.stop = vec!["END".into()];
        dispatcher(registry)
            .generate(2, &context, request, tenant_work())
            .await
            .unwrap();
        let requests = worker.requests();
        let InvocationInput::Chat { parameters, .. } = &requests[0].input;
        assert_eq!(parameters.temperature, Some(1.0));
        assert_eq!(parameters.top_p, Some(0.95));
        assert_eq!(parameters.stop, ["END"]);
    }

    /// DP-F2: every control a request sets reaches a worker that advertises
    /// it, with the negotiated version and the D8 context policy.
    #[tokio::test]
    async fn worker_receives_every_control_it_advertises() {
        let worker =
            ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        dispatcher(registry)
            .generate(2, &context, every_control(), tenant_work())
            .await
            .unwrap();
        let requests = worker.requests();
        assert_eq!(requests[0].schema_version, PROTOCOL_V1);
        let InvocationInput::Chat { parameters, .. } = &requests[0].input;
        assert_eq!(
            *parameters,
            WorkerChatParameters {
                temperature: Some(1.0),
                top_p: Some(0.95),
                seed: Some(7),
                stop: vec!["END".into()],
                top_k: Some(9),
                repetition_penalty: Some(1.25),
                presence_penalty: Some(0.5),
                logprobs: Some(2),
                response_format: Some(ChatResponseFormat::JsonObject),
                context: Some(ChatContextPolicy {
                    max_tokens: Some(32),
                    truncation: WorkerChatTruncation::Disabled,
                }),
                tools: vec![serde_json::json!({"type": "function", "function": {"name": "now"}})],
            }
        );
    }

    fn every_control() -> ChatExecutionRequest {
        let mut request = request();
        request.seed = Some(7);
        request.stop = vec!["END".into()];
        request.top_k = Some(9);
        request.repetition_penalty = Some(1.25);
        request.presence_penalty = Some(0.5);
        request.logprobs = Some(true);
        request.top_logprobs = Some(2);
        request.response_format_json_object = true;
        request.chat_config.tools =
            vec![serde_json::json!({"type": "function", "function": {"name": "now"}})];
        request
    }

    /// A control goes only to a worker that advertises it: a mixed fleet
    /// routes it to the newer worker, and a fleet without one refuses it
    /// with a 400 instead of letting a worker drop it. A plain request still
    /// runs on an older worker, at its minor and without the context policy.
    #[tokio::test]
    async fn controls_route_only_to_workers_that_advertise_them() {
        let older =
            ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
        let newer =
            ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register_minor3(&registry, "worker-a", "inc-a", client(&older.endpoint()));
        register(&registry, "worker-b", "inc-b", client(&newer.endpoint()), 1);
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        dispatcher(registry)
            .generate(2, &context, every_control(), tenant_work())
            .await
            .unwrap();
        assert!(older.requests().is_empty());
        assert_eq!(newer.requests().len(), 1);

        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register_minor3(&registry, "worker-a", "inc-a", client(&older.endpoint()));
        let dispatcher = dispatcher(registry);
        let cases: [(&str, fn(&mut ChatExecutionRequest)); 7] = [
            ("seed", |request| request.seed = Some(7)),
            ("stop", |request| request.stop = vec!["END".into()]),
            ("top_k", |request| request.top_k = Some(9)),
            ("presence_penalty", |request| {
                request.presence_penalty = Some(0.5)
            }),
            ("logprobs", |request| request.logprobs = Some(true)),
            ("response_format", |request| {
                request.response_format_json_object = true
            }),
            ("tools", |request| {
                request.chat_config.tools = vec![serde_json::json!({"type": "function"})]
            }),
        ];
        for (parameter, apply) in cases {
            let mut unsupported = request();
            apply(&mut unsupported);
            let error = dispatcher
                .generate(2, &context, unsupported, tenant_work())
                .await
                .unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_REQUEST, "{parameter}");
            assert_eq!(
                error.detail.and_then(|detail| detail.param),
                Some(parameter)
            );
            assert_eq!(
                error.detail.and_then(|detail| detail.code),
                Some("unsupported_parameter")
            );
        }
        assert!(older.requests().is_empty());

        // An explicit `logprobs: false` asks for nothing the worker lacks.
        let mut plain = request();
        plain.logprobs = Some(false);
        dispatcher
            .generate(2, &context, plain, tenant_work())
            .await
            .unwrap();
        let requests = older.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].schema_version, SchemaVersion::new(1, 3));
        let InvocationInput::Chat { parameters, .. } = &requests[0].input;
        assert_eq!(parameters.context, None);
        assert_eq!(parameters.temperature, Some(1.0));
    }

    #[tokio::test]
    async fn unknown_admission_is_a_busy_worker_not_a_circuit_strike() {
        let worker =
            ScriptedWorker::spawn(ScriptedResponse::AdmissionUnknown, Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let dispatcher = dispatcher(registry);
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

        // More slow admissions than the circuit's failure threshold (3).
        for _ in 0..4 {
            let error = dispatcher
                .generate(2, &context, request(), tenant_work())
                .await
                .unwrap_err();
            assert_eq!(
                error.status,
                StatusCode::SERVICE_UNAVAILABLE,
                "{}",
                error.message
            );
        }
        // An unknown outcome is never retried elsewhere, and the circuit
        // stays closed, so every request still reached the worker.
        assert_eq!(worker.requests().len(), 4);
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
            .stream(2, &context, request(), tenant_work())
            .await
            .unwrap();
        assert!(matches!(
            events.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        assert!(matches!(
            events.recv().await,
            Some(ChatStreamEvent::Delta { text, .. }) if text == "from-b"
        ));
        assert!(matches!(
            events.recv().await,
            Some(ChatStreamEvent::Completed(_))
        ));
        assert_eq!(worker_b.requests().len(), 1);
    }

    /// A worker burst three times the relay channel's capacity.
    const BURST_DELTAS: usize = 192;

    #[tokio::test]
    async fn paused_public_reader_receives_every_worker_delta() {
        let count = BURST_DELTAS;
        let worker =
            ScriptedWorker::spawn(ScriptedResponse::ManyDeltas(count), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let mut public_events = dispatcher(registry)
            .stream(
                2,
                &RequestContext::new("test-request".into(), Principal::local_anonymous()),
                request(),
                tenant_work(),
            )
            .await
            .unwrap();

        // The worker's burst outruns both relay channels while the reader
        // pauses for less than the 100 ms slow-consumer timeout.
        tokio::time::sleep(Duration::from_millis(30)).await;
        let mut text = String::new();
        let terminal = loop {
            match public_events.recv().await {
                Some(ChatStreamEvent::Delta { text: delta, .. }) => text.push_str(&delta),
                Some(ChatStreamEvent::Started) => {}
                terminal => break terminal,
            }
        };
        assert!(
            matches!(terminal, Some(ChatStreamEvent::Completed(_))),
            "{terminal:?}"
        );
        assert_eq!(text, "x".repeat(count));
    }

    #[tokio::test]
    async fn non_draining_public_stream_cannot_retain_registry_dispatch() {
        let worker =
            ScriptedWorker::spawn(ScriptedResponse::ManyDeltas(BURST_DELTAS), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            max_local_dispatches: 1,
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let dispatcher = dispatcher(registry.clone());
        let selection = dispatcher.selection_request(&request(), true).unwrap();

        let mut public_events = dispatcher
            .stream(
                2,
                &RequestContext::new("test-request".into(), Principal::local_anonymous()),
                request(),
                tenant_work(),
            )
            .await
            .unwrap();
        let replacement = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match registry.select_and_reserve(&selection) {
                    Ok(selected) => break selected,
                    Err(WorkerRegistryError::LocalDispatchLimitReached) => {
                        tokio::task::yield_now().await;
                    }
                    Err(error) => panic!("unexpected replacement selection failure: {error}"),
                }
            }
        })
        .await
        .expect("slow-consumer forwarding timeout must release dispatch ownership");
        drop(replacement);

        let mut saw_completion = false;
        tokio::time::timeout(Duration::from_secs(1), async {
            while let Some(event) = public_events.recv().await {
                saw_completion |= matches!(event, ChatStreamEvent::Completed(_));
            }
        })
        .await
        .expect("timed-out bridge must close its public event channel");
        assert!(
            !saw_completion,
            "a truncated slow-consumer stream must never fabricate completion"
        );
        // `render_chat_stream` maps this non-terminal channel closure to an
        // explicit SSE error before [DONE]; its dedicated route-level test
        // covers that public encoding contract.
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
            .generate(2, &context, request(), tenant_work())
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

    async fn fleet_coordinator() -> (
        Arc<crate::app::fleet_coordinator::FleetCoordinator>,
        tempfile::TempDir,
    ) {
        let root = tempfile::tempdir().unwrap();
        let store = FleetStore::new(StoreDatabase::new(
            root.path().join("fleet.sqlite3"),
        ));
        (
            Arc::new(
                crate::app::fleet_coordinator::FleetCoordinator::new(
                    store,
                    "gateway-test".to_string(),
                )
                .opened()
                .await,
            ),
            root,
        )
    }

    #[tokio::test]
    async fn fleet_claim_is_published_and_released_across_generate() {
        let worker = ScriptedWorker::spawn(
            ScriptedResponse::Success("from-fleet".into()),
            Duration::ZERO,
        )
        .await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let (coordinator, _root) = fleet_coordinator().await;
        let store = coordinator.store();
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        let generation = dispatcher(registry)
            .with_fleet_coordinator(Arc::clone(&coordinator))
            .generate(2, &context, request(), tenant_work())
            .await
            .unwrap();
        assert_eq!(generation.text, "from-fleet");
        for _ in 0..200 {
            if store.count_live_fleet_claims("worker-a").await.unwrap() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            store.count_live_fleet_claims("worker-a").await.unwrap(),
            0,
            "the dispatch claim guard must release its cluster claim when the request completes"
        );
    }

    #[tokio::test]
    async fn accepted_work_releases_its_fleet_claim_so_free_credits_stay_dispatchable() {
        // Four credits, two accepted in-flight streams from gateway A: the
        // worker reports two free credits, which must not be shadowed again by
        // the accepted streams' claims, for A or for its peer gateway B.
        let worker = ScriptedWorker::spawn(ScriptedResponse::AcceptThenHold, Duration::ZERO).await;
        let root = tempfile::tempdir().unwrap();
        let fleet_db = root.path().join("fleet.sqlite3");
        let gateway = |gateway_id: &str| {
            let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
            register_with_credits(
                &registry,
                "worker-a",
                "inc-a",
                client(&worker.endpoint()),
                1,
                4,
            );
            let coordinator = crate::app::fleet_coordinator::FleetCoordinator::new(
                FleetStore::new(StoreDatabase::new(fleet_db.clone())),
                gateway_id.to_string(),
            );
            (registry, coordinator)
        };
        let open = |(registry, coordinator): (
            WorkerRegistry,
            crate::app::fleet_coordinator::FleetCoordinator,
        )| async move {
            let coordinator = Arc::new(coordinator.opened().await);
            let dispatcher =
                dispatcher(registry.clone()).with_fleet_coordinator(Arc::clone(&coordinator));
            (registry, coordinator, dispatcher)
        };
        let (registry_a, coordinator_a, dispatcher_a) = open(gateway("gateway-a")).await;
        let (registry_b, coordinator_b, dispatcher_b) = open(gateway("gateway-b")).await;
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        let mut held = Vec::new();
        for _ in 0..2 {
            held.push(
                dispatcher_a
                    .stream(2, &context, request(), tenant_work())
                    .await
                    .unwrap(),
            );
        }
        // Guards release on a spawned task; give a released claim time to
        // land before the polls sample the store. A claim held to the end of
        // the stream never lands, and the dispatches below then shed.
        let store = coordinator_a.store();
        for _ in 0..200 {
            if store.count_live_fleet_claims("worker-a").await.unwrap() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // Both gateways' next status polls report the two accepted streams.
        let status = worker_status("worker-a", "inc-a", 2, 4, 2);
        for (registry, coordinator) in
            [(&registry_a, &coordinator_a), (&registry_b, &coordinator_b)]
        {
            registry.observe_status(status.clone()).unwrap();
            coordinator.refresh_peer_claims(&status).await;
        }

        for (name, dispatcher) in [
            ("gateway A", &dispatcher_a),
            ("peer gateway B", &dispatcher_b),
        ] {
            let next = dispatcher
                .stream(2, &context, request(), tenant_work())
                .await;
            assert!(
                next.is_ok(),
                "{name} must still dispatch to the two free credits: {:?}",
                next.err()
            );
            held.push(next.unwrap());
        }
        assert_eq!(worker.requests().len(), 4);
    }

    #[tokio::test]
    async fn hung_fleet_store_costs_at_most_the_claim_timeout() {
        let worker = ScriptedWorker::spawn(
            ScriptedResponse::Success("from-fleet".into()),
            Duration::ZERO,
        )
        .await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let root = tempfile::tempdir().unwrap();
        let mut store = FleetStore::new(StoreDatabase::new(
            root.path().join("fleet.sqlite3"),
        ));
        let faults = Arc::new(crate::gateway_fleet::FleetFaults::default());
        store.set_faults_for_test(Arc::clone(&faults));
        faults.hang_fleet_calls();
        let coordinator = Arc::new(
            crate::app::fleet_coordinator::FleetCoordinator::new(store, "gateway-test".into())
                .with_claim_timeout(Duration::from_millis(50))
                .opened()
                .await,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        // The worker stays the admission arbiter: a store that never answers
        // the claim must not hold the request on its time-to-first-token.
        let generation = tokio::time::timeout(
            Duration::from_secs(2),
            dispatcher(registry)
                .with_fleet_coordinator(coordinator)
                .generate(2, &context, request(), tenant_work()),
        )
        .await
        .expect("a hung coordination store must not stall dispatch")
        .unwrap();
        assert_eq!(generation.text, "from-fleet");
        faults.answer_fleet_calls();
    }

    #[tokio::test]
    async fn fleet_claim_loss_still_invokes_and_succeeds() {
        let worker = ScriptedWorker::spawn(
            ScriptedResponse::Success("from-fleet".into()),
            Duration::ZERO,
        )
        .await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
        );
        let (coordinator, _root) = fleet_coordinator().await;
        // A peer holds the only observable credit, but this gateway's cached
        // snapshot is stale (empty): selection proceeds, the claim loses, and
        // the worker arbitrates the invocation to success.
        let peer = coordinator
            .claim(
                &crate::worker_registry::WorkerInstanceKey {
                    worker_id: id::<WorkerId>("worker-a"),
                    incarnation_id: id::<IncarnationId>("inc-a"),
                },
                1,
            )
            .await
            .expect("peer claim");
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
        let generation = dispatcher(registry)
            .with_fleet_coordinator(Arc::clone(&coordinator))
            .generate(2, &context, request(), tenant_work())
            .await
            .unwrap();
        assert_eq!(generation.text, "from-fleet");
        drop(peer);
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
                .generate(2, &context, request(), tenant_work())
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
            .generate(2, &context, request(), tenant_work())
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert!(worker_c.requests().is_empty());
    }

    fn conversation_turn(system: &str, turns: &[(&str, Option<&str>)]) -> ChatExecutionRequest {
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            content: system.into(),
        }];
        for (user, assistant) in turns {
            messages.push(ChatMessage {
                role: ChatRole::User,
                content: (*user).into(),
            });
            if let Some(assistant) = assistant {
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: (*assistant).into(),
                });
            }
        }
        ChatExecutionRequest {
            variant: ModelVariant::Qwen354BGguf,
            messages,
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: Vec::new(),
            logprobs: None,
            top_logprobs: None,
            response_format_json_object: false,
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        }
    }

    fn turn_one() -> ChatExecutionRequest {
        conversation_turn(
            "You are a terse bearologist.",
            &[("Tell me about bears.", None)],
        )
    }

    fn turn_two() -> ChatExecutionRequest {
        conversation_turn(
            "You are a terse bearologist.",
            &[("Tell me about bears.", Some("Bears are large mammals."))],
        )
    }

    async fn two_ready_workers(
        routing: RoutingConfig,
    ) -> (ScriptedWorker, ScriptedWorker, WorkerRegistry) {
        let worker_a =
            ScriptedWorker::spawn(ScriptedResponse::Success("from-a".into()), Duration::ZERO).await;
        let worker_b =
            ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            routing,
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        for (name, incarnation, worker) in [
            ("worker-a", "inc-a", &worker_a),
            ("worker-b", "inc-b", &worker_b),
        ] {
            register_with_credits(
                &registry,
                name,
                incarnation,
                client(&worker.endpoint()),
                1,
                4,
            );
        }
        (worker_a, worker_b, registry)
    }

    fn small_blocks() -> Option<PrefixAffinityConfig> {
        Some(PrefixAffinityConfig {
            block_bytes: 16,
            ..PrefixAffinityConfig::default()
        })
    }

    #[tokio::test]
    async fn conversation_turns_follow_the_worker_holding_their_prefix() {
        let (worker_a, worker_b, registry) = two_ready_workers(RoutingConfig::default()).await;
        let dispatcher = dispatcher_with_prefix(registry.clone(), small_blocks());
        let context = RequestContext::new("prefix-request".into(), Principal::local_anonymous());

        // worker-a is busy, so turn one goes to the idle worker-b.
        let busy_a = registry
            .select_and_reserve(&dispatcher.selection_request(&turn_one(), false).unwrap())
            .unwrap();
        assert_eq!(busy_a.key.worker_id.as_str(), "worker-a");
        dispatcher
            .generate(2, &context, turn_one(), tenant_work())
            .await
            .unwrap();
        assert_eq!(worker_b.requests().len(), 1);
        drop(busy_a);

        // Both are idle again and worker-a holds the identity tie-break, but
        // turn two shares turn one's prefix, which worker-b holds.
        dispatcher
            .generate(2, &context, turn_two(), tenant_work())
            .await
            .unwrap();
        assert_eq!(worker_b.requests().len(), 2);
        assert!(worker_a.requests().is_empty());

        // A different conversation has no held prefix: load decides.
        dispatcher
            .generate(
                2,
                &context,
                conversation_turn("Ornithologist on duty.", &[("Owls?", None)]),
                tenant_work(),
            )
            .await
            .unwrap();
        assert_eq!(worker_a.requests().len(), 1);
    }

    #[tokio::test]
    async fn prefix_affinity_spills_to_the_least_loaded_worker_past_the_balance_threshold() {
        for (threshold, expected) in [(1, "worker-a"), (0, "worker-b")] {
            let (worker_a, worker_b, registry) = two_ready_workers(RoutingConfig {
                prefix_balance_threshold: threshold,
                ..RoutingConfig::default()
            })
            .await;
            let dispatcher = dispatcher_with_prefix(registry.clone(), small_blocks());
            let context = RequestContext::new("prefix-spill".into(), Principal::local_anonymous());
            dispatcher
                .generate(2, &context, turn_one(), tenant_work())
                .await
                .unwrap();
            assert_eq!(worker_a.requests().len(), 1);

            // One more request in flight on the prefix holder.
            let busy_a = registry
                .select_and_reserve(&dispatcher.selection_request(&turn_one(), false).unwrap())
                .unwrap();
            assert_eq!(busy_a.key.worker_id.as_str(), "worker-a");
            dispatcher
                .generate(2, &context, turn_two(), tenant_work())
                .await
                .unwrap();
            let served_by = if worker_a.requests().len() == 2 {
                "worker-a"
            } else {
                assert_eq!(worker_b.requests().len(), 1);
                "worker-b"
            };
            assert_eq!(served_by, expected, "threshold {threshold}");
        }
    }

    #[tokio::test]
    async fn prefix_recorded_through_failover_steers_the_next_turn() {
        let worker_a = ScriptedWorker::spawn(
            ScriptedResponse::Reject(RejectionCode::CapacityExhausted),
            Duration::ZERO,
        )
        .await;
        let worker_b =
            ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        // worker-a refuses for capacity, so turn one fails over to worker-b,
        // which must record the prefix through the alternate-dispatch path.
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
        let dispatcher = dispatcher_with_prefix(registry, small_blocks());
        let context = RequestContext::new("prefix-failover".into(), Principal::local_anonymous());

        dispatcher
            .generate(2, &context, turn_one(), tenant_work())
            .await
            .unwrap();
        assert_eq!(worker_a.requests().len(), 1);
        assert_eq!(worker_b.requests().len(), 1);

        // The second turn goes straight to worker-b instead of first trying
        // the identity-preferred worker-a again.
        dispatcher
            .generate(2, &context, turn_two(), tenant_work())
            .await
            .unwrap();
        assert_eq!(worker_a.requests().len(), 1);
        assert_eq!(worker_b.requests().len(), 2);
    }
}

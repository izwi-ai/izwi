//! Multi-worker routing for the migrated public chat slice.
//!
//! The registry narrows candidates using fresh, receiver-clock status, but the
//! chosen worker remains authoritative for admission. A dispatcher may make
//! one alternate attempt only when the first attempt is provably unaccepted;
//! uncertain acceptance and accepted execution never fail over.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use izwi_model_catalog::{ChatGeneration, ModelVariant};
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
    pub fn with_shutdown(mut self, shutdown: tokio::sync::watch::Receiver<bool>) -> Self {
        self.shutdown = Some(shutdown);
        self
    }

    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    pub const fn max_output_tokens(&self) -> u32 {
        self.config.max_output_tokens
    }

    /// The model this deployment serves under its public name.
    pub const fn public_model(&self) -> ModelVariant {
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

    pub async fn generate(
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
    pub async fn stream(
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
#[path = "remote_chat_dispatch_tests.rs"]
mod tests;

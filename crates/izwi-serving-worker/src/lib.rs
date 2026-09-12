//! Private, versioned HTTP worker boundary for Izwi inference runtimes.
//!
//! The HTTP layer owns bounded request parsing, attempt fencing, and transport-level
//! concurrency. An [`InvocationExecutor`] owns authoritative runtime admission. An
//! invocation is never advertised as accepted until both layers have admitted it.

use async_stream::stream;
use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use izwi_serving_protocol::*;
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::sync::{mpsc, watch, OwnedRwLockReadGuard, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::time::Instant;

mod runtime;

pub use runtime::{warm_up_chat_runtime, RuntimeChatExecutor};

pub const DEFAULT_MAX_REQUEST_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MAX_RETAINED_ATTEMPTS: usize = 1024;
pub const DEFAULT_ATTEMPT_RETENTION: Duration = Duration::from_secs(300);
pub const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 4;
pub const DEFAULT_MAX_EVENT_BYTES: usize = 1024 * 1024;
const EVENT_ENVELOPE_ALLOWANCE: usize = 1024;

/// Immutable worker identity and bounded local resource policy.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub descriptor: WorkerDescriptor,
    pub deployment: LoadedDeployment,
    pub credentials: ServiceCredentials,
    pub max_active_invocations: usize,
    pub max_request_bytes: usize,
    pub max_retained_attempts: usize,
    pub attempt_retention: Duration,
    pub event_channel_capacity: usize,
    pub max_event_bytes: usize,
}

impl WorkerConfig {
    pub fn validate(&self) -> Result<(), WorkerConfigError> {
        if self.descriptor.schema_version.major != PROTOCOL_V1.major
            || !self
                .descriptor
                .supported_protocol_versions
                .iter()
                .any(|version| version.major == PROTOCOL_V1.major)
        {
            return Err(WorkerConfigError::ProtocolVersion);
        }
        if self.descriptor.worker_id.as_str().is_empty()
            || self.descriptor.incarnation_id.as_str().is_empty()
        {
            return Err(WorkerConfigError::Identity);
        }
        if self.descriptor.assignment.backend() != self.deployment.backend {
            return Err(WorkerConfigError::BackendMismatch);
        }
        if self.deployment.task != TaskKind::Chat
            || self.deployment.capability.task != TaskKind::Chat
            || !self
                .deployment
                .capability
                .accepted_input_formats
                .contains(&InputFormat::ChatMessages)
            || !self
                .deployment
                .capability
                .output_formats
                .contains(&OutputFormat::Text)
        {
            return Err(WorkerConfigError::UnsupportedDeployment);
        }
        if self.max_active_invocations == 0 {
            return Err(WorkerConfigError::ZeroActiveCapacity);
        }
        if self.max_request_bytes == 0 {
            return Err(WorkerConfigError::ZeroRequestLimit);
        }
        if self.max_retained_attempts < self.max_active_invocations {
            return Err(WorkerConfigError::AttemptRetention);
        }
        if self.attempt_retention < Duration::from_secs(1)
            || self.attempt_retention > Duration::from_secs(86_400)
        {
            return Err(WorkerConfigError::AttemptRetentionWindow);
        }
        // Accepted plus one text delta plus one terminal event must never block
        // completion on a slow or disconnected transport consumer.
        if self.event_channel_capacity < 3 {
            return Err(WorkerConfigError::EventChannelCapacity);
        }
        if self.max_event_bytes < EVENT_ENVELOPE_ALLOWANCE * 2 {
            return Err(WorkerConfigError::EventLimit);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerConfigError {
    #[error("worker must advertise private protocol v1")]
    ProtocolVersion,
    #[error("worker identity fields must be non-empty")]
    Identity,
    #[error("device assignment and deployment backend differ")]
    BackendMismatch,
    #[error("this worker slice supports exactly one chat deployment")]
    UnsupportedDeployment,
    #[error("max_active_invocations must be non-zero")]
    ZeroActiveCapacity,
    #[error("max_request_bytes must be non-zero")]
    ZeroRequestLimit,
    #[error("attempt retention must cover every active invocation")]
    AttemptRetention,
    #[error("attempt retention window must be between one second and one day")]
    AttemptRetentionWindow,
    #[error("event channel capacity must hold accepted, delta, and terminal events")]
    EventChannelCapacity,
    #[error("max_event_bytes must be at least 2048")]
    EventLimit,
}

/// A rejection returned before runtime ownership has been accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionFailure {
    pub code: RejectionCode,
    pub message: String,
    pub retry_after_ms: Option<u64>,
}

impl AdmissionFailure {
    pub fn new(code: RejectionCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retry_after_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFailure {
    pub code: InvocationErrorCode,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionEvent {
    TextDelta(String),
    Completed {
        /// Non-streaming runtimes return their one bounded text value here.
        /// Streaming adapters leave this empty after forwarding deltas.
        text: Option<String>,
        finish_reason: FinishReason,
        input_tokens: u64,
        output_tokens: u64,
    },
    Failed(ExecutionFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTeardown {
    Completed,
    Cancelled,
    Failed,
    /// Physical teardown could not be proven; capacity must remain held.
    Unconfirmed,
}

/// Opaque admitted runtime execution retained through confirmed teardown.
#[async_trait]
pub trait AdmittedExecution: Send + 'static {
    /// Return the next bounded output event. `None` is not teardown proof.
    async fn next_event(&mut self) -> Option<ExecutionEvent>;

    /// Cooperatively signal cancellation without claiming execution stopped.
    fn request_cancel(&self);

    /// Resolve only after native completion or exact-session cleanup.
    async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown;
}

/// Runtime ownership obtained atomically before the HTTP worker emits Accepted.
pub struct AdmittedInvocation {
    execution: Box<dyn AdmittedExecution>,
}

impl AdmittedInvocation {
    pub fn new(execution: Box<dyn AdmittedExecution>) -> Self {
        Self { execution }
    }

    fn into_execution(self) -> Box<dyn AdmittedExecution> {
        self.execution
    }
}

/// Adapter from the private protocol into an inference runtime.
///
/// `admit` must return only after the runtime owns all authoritative request,
/// model-residency, scheduler, and physical-capacity leases needed by the
/// invocation. The returned execution must retain ownership until native
/// execution has completed or teardown has been confirmed. The worker never
/// drops an in-flight admission future merely because its HTTP wait expires.
#[async_trait]
pub trait InvocationExecutor: Send + Sync + 'static {
    async fn admit(
        &self,
        request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure>;
}

#[derive(Clone)]
struct AttemptRecord {
    identity: AttemptIdentity,
    digest: RequestDigest,
    state: AttemptState,
    last_sequence: Option<u64>,
    cancel_requested: bool,
    cancel: Option<watch::Sender<bool>>,
    evict_after: Option<Instant>,
}

#[derive(Default)]
struct AttemptTable {
    records: HashMap<AttemptId, AttemptRecord>,
    order: VecDeque<AttemptId>,
}

struct WorkerState<E> {
    config: WorkerConfig,
    executor: Arc<E>,
    capacity: Arc<Semaphore>,
    admission_gate: Arc<RwLock<()>>,
    attempts: Mutex<AttemptTable>,
    status_sequence: AtomicU64,
    draining: AtomicBool,
}

impl<E> WorkerState<E> {
    fn remove_record(table: &mut AttemptTable, attempt_id: &AttemptId) {
        table.records.remove(attempt_id);
        if let Some(index) = table.order.iter().position(|id| id == attempt_id) {
            table.order.remove(index);
        }
    }

    fn record_is_expired(record: &AttemptRecord, now: Instant) -> bool {
        record.evict_after.is_some_and(|deadline| deadline <= now)
    }

    fn evict_expired_record(&self, table: &mut AttemptTable, now: Instant) -> bool {
        let Some(index) = table.order.iter().position(|attempt_id| {
            table
                .records
                .get(attempt_id)
                .and_then(|record| record.evict_after)
                .is_some_and(|deadline| deadline <= now)
        }) else {
            return false;
        };
        let evicted = table.order.remove(index).expect("known retention index");
        table.records.remove(&evicted);
        true
    }

    fn authenticate(&self, headers: &HeaderMap) -> bool {
        let bearer = headers
            .get(SERVICE_AUTHORIZATION_HEADER)
            .and_then(|value| value.to_str().ok());
        let credential = headers
            .get(SERVICE_CREDENTIAL_ID_HEADER)
            .and_then(|value| value.to_str().ok());
        let presented_token = bearer.and_then(|value| {
            value
                .strip_prefix(SERVICE_AUTH_SCHEME)
                .and_then(|value| value.strip_prefix(' '))
        });
        credential == Some(self.config.credentials.credential_id.as_str())
            && presented_token.is_some_and(|token| {
                self.config
                    .credentials
                    .bearer_token
                    .matches_presented(token)
            })
    }

    fn reserve_attempt(&self, request: &InvocationRequest) -> ReserveAttempt {
        let mut table = self.attempts.lock().expect("worker attempt table poisoned");
        if table
            .records
            .get(&request.attempt_id)
            .is_some_and(|record| Self::record_is_expired(record, Instant::now()))
        {
            Self::remove_record(&mut table, &request.attempt_id);
        }
        if let Some(existing) = table.records.get(&request.attempt_id) {
            return if existing.identity == AttemptIdentity::from(request)
                && existing.digest == request.request_digest
            {
                ReserveAttempt::AlreadyOwned
            } else {
                ReserveAttempt::Conflict
            };
        }
        while table.records.len() >= self.config.max_retained_attempts {
            if !self.evict_expired_record(&mut table, Instant::now()) {
                return ReserveAttempt::Full;
            }
        }
        table.order.push_back(request.attempt_id.clone());
        table.records.insert(
            request.attempt_id.clone(),
            AttemptRecord {
                identity: AttemptIdentity::from(request),
                digest: request.request_digest.clone(),
                state: AttemptState::Queued,
                last_sequence: None,
                cancel_requested: false,
                cancel: None,
                evict_after: None,
            },
        );
        ReserveAttempt::Reserved
    }

    fn remove_reservation(&self, attempt_id: &AttemptId) {
        let mut table = self.attempts.lock().expect("worker attempt table poisoned");
        Self::remove_record(&mut table, attempt_id);
    }

    fn install_admission(&self, attempt_id: &AttemptId, cancel: watch::Sender<bool>) -> bool {
        let mut table = self.attempts.lock().expect("worker attempt table poisoned");
        let record = table
            .records
            .get_mut(attempt_id)
            .expect("reserved attempt must exist through admission");
        record.state = AttemptState::Admitted;
        record.last_sequence = Some(0);
        record.cancel = Some(cancel);
        record.cancel_requested
    }

    fn update_attempt(&self, attempt_id: &AttemptId, state: AttemptState, sequence: Option<u64>) {
        let mut table = self.attempts.lock().expect("worker attempt table poisoned");
        if let Some(record) = table.records.get_mut(attempt_id) {
            record.state = state;
            if let Some(sequence) = sequence {
                record.last_sequence = Some(sequence);
            }
            if state.is_terminal() {
                record.cancel = None;
                record.evict_after = Instant::now().checked_add(self.config.attempt_retention);
            }
        }
    }

    fn mark_cancellation_requested(&self, attempt_id: &AttemptId) {
        let mut table = self.attempts.lock().expect("worker attempt table poisoned");
        if let Some(record) = table.records.get_mut(attempt_id) {
            record.cancel_requested = true;
            record.state = AttemptState::CancellationRequested;
        }
    }

    fn insert_cancel_tombstone(&self, identity: AttemptIdentity) {
        let mut table = self.attempts.lock().expect("worker attempt table poisoned");
        if table.records.contains_key(&identity.attempt_id) {
            return;
        }
        while table.records.len() >= self.config.max_retained_attempts {
            if !self.evict_expired_record(&mut table, Instant::now()) {
                return;
            }
        }
        let attempt_id = identity.attempt_id.clone();
        table.order.push_back(attempt_id.clone());
        table.records.insert(
            attempt_id,
            AttemptRecord {
                identity,
                digest: RequestDigest::new("cancel-tombstone").expect("static identity"),
                state: AttemptState::CancellationRequested,
                last_sequence: None,
                cancel_requested: true,
                cancel: None,
                evict_after: Instant::now().checked_add(self.config.attempt_retention),
            },
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReserveAttempt {
    Reserved,
    AlreadyOwned,
    Conflict,
    Full,
}

struct AdmissionHandoff {
    result: Result<AdmittedInvocation, AdmissionFailure>,
    permit: OwnedSemaphorePermit,
    admission_guard: OwnedRwLockReadGuard<()>,
}

#[derive(Default)]
struct AdmissionSlot {
    handoff: Option<AdmissionHandoff>,
    abandoned: bool,
}

/// Cloneable owner of the worker router and lifecycle state.
pub struct WorkerService<E> {
    state: Arc<WorkerState<E>>,
}

impl<E> Clone for WorkerService<E> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<E: InvocationExecutor> WorkerService<E> {
    pub fn new(config: WorkerConfig, executor: E) -> Result<Self, WorkerConfigError> {
        config.validate()?;
        let capacity = Arc::new(Semaphore::new(config.max_active_invocations));
        Ok(Self {
            state: Arc::new(WorkerState {
                config,
                executor: Arc::new(executor),
                capacity,
                admission_gate: Arc::new(RwLock::new(())),
                attempts: Mutex::new(AttemptTable::default()),
                status_sequence: AtomicU64::new(0),
                draining: AtomicBool::new(false),
            }),
        })
    }

    pub fn router(&self) -> Router {
        let max_request_bytes = self.state.config.max_request_bytes;
        Router::new()
            .route(WORKER_DESCRIPTOR_PATH, get(descriptor::<E>))
            .route(WORKER_STATUS_PATH, get(status::<E>))
            .route(INVOCATIONS_PATH, post(invoke::<E>))
            .route(
                &format!("{INVOCATIONS_PATH}/{{attempt_id}}"),
                get(query_attempt::<E>),
            )
            .route(
                &format!("{INVOCATIONS_PATH}/{{attempt_id}}/cancel"),
                post(cancel_attempt::<E>),
            )
            .layer(DefaultBodyLimit::max(max_request_bytes))
            .with_state(Arc::clone(&self.state))
    }

    pub async fn begin_draining(&self) {
        let _gate = self.state.admission_gate.write().await;
        self.state.draining.store(true, Ordering::Release);
    }

    pub fn active_invocations(&self) -> usize {
        self.state.config.max_active_invocations - self.state.capacity.available_permits()
    }
}

async fn descriptor<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Json(state.config.descriptor.clone()).into_response()
}

async fn status<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let active = state.config.max_active_invocations - state.capacity.available_permits();
    let draining = state.draining.load(Ordering::Acquire);
    let mut deployment = state.config.deployment.clone();
    if draining {
        deployment.readiness = ModelReadiness::Draining;
    }
    Json(WorkerStatus {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.descriptor.worker_id.clone(),
        node_id: state.config.descriptor.node_id.clone(),
        incarnation_id: state.config.descriptor.incarnation_id.clone(),
        status_sequence: state.status_sequence.fetch_add(1, Ordering::Relaxed) + 1,
        process_state: if draining {
            WorkerProcessState::Draining
        } else {
            WorkerProcessState::Running
        },
        deployments: vec![deployment],
        capacity: CapacitySnapshot {
            max_active_invocations: state.config.max_active_invocations as u32,
            active_invocations: active as u32,
            max_queued_invocations: 0,
            queued_invocations: 0,
            max_sessions: 0,
            reserved_sessions: 0,
            available_admission_credits: state.capacity.available_permits() as u32,
            outstanding_cost_units: active as u64,
        },
    })
    .into_response()
}

async fn invoke<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let received_at = Instant::now();
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let request: InvocationRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    if request.schema_version.major != PROTOCOL_V1.major {
        return rejection(
            &request,
            StatusCode::CONFLICT,
            RejectionCode::UnsupportedProtocolVersion,
            "unsupported protocol major version",
            None,
        );
    }
    if let Err(error) = request.validate() {
        return rejection(
            &request,
            StatusCode::BAD_REQUEST,
            RejectionCode::InvalidRequest,
            error.to_string(),
            None,
        );
    }
    if !request
        .caller
        .permitted_actions
        .contains(&PermittedAction::Invoke)
    {
        return rejection(
            &request,
            StatusCode::FORBIDDEN,
            RejectionCode::PolicyDenied,
            "gateway caller context does not permit invocation",
            None,
        );
    }
    if state.draining.load(Ordering::Acquire) {
        return rejection(
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::WorkerDraining,
            "worker is draining",
            None,
        );
    }
    if request.expected_worker_incarnation != state.config.descriptor.incarnation_id {
        return rejection(
            &request,
            StatusCode::CONFLICT,
            RejectionCode::WrongWorkerIncarnation,
            "worker incarnation changed",
            None,
        );
    }
    if request.deployment_id != state.config.deployment.deployment_id {
        return rejection(
            &request,
            StatusCode::NOT_FOUND,
            RejectionCode::UnknownDeployment,
            "deployment is not loaded",
            None,
        );
    }
    if request.expected_model_generation != state.config.deployment.model_generation {
        return rejection(
            &request,
            StatusCode::CONFLICT,
            RejectionCode::WrongModelGeneration,
            "model generation changed",
            None,
        );
    }
    if request.task != TaskKind::Chat || request.input.task() != TaskKind::Chat {
        return rejection(
            &request,
            StatusCode::UNPROCESSABLE_ENTITY,
            RejectionCode::IncompatibleTask,
            "task is incompatible with deployment",
            None,
        );
    }
    if request.service_class == ServiceClass::Realtime
        && !state.config.deployment.capability.realtime
    {
        return rejection(
            &request,
            StatusCode::UNPROCESSABLE_ENTITY,
            RejectionCode::IncompatibleTask,
            "deployment does not support realtime execution",
            None,
        );
    }
    if state.config.deployment.readiness != ModelReadiness::Ready {
        return rejection(
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::ModelNotReady,
            "deployment is not ready",
            None,
        );
    }
    let input_bytes = match &request.input {
        InvocationInput::Chat { input, .. } => input
            .messages
            .iter()
            .try_fold(0u64, |total, message| {
                total.checked_add(message.content.len() as u64)
            })
            .unwrap_or(u64::MAX),
    };
    if input_bytes > state.config.deployment.capability.max_input_bytes
        || usize::try_from(request.output_limits.max_bytes).map_or(true, |limit| {
            limit.saturating_add(EVENT_ENVELOPE_ALLOWANCE) > state.config.max_event_bytes
        })
        || state
            .config
            .deployment
            .capability
            .max_output_tokens
            .is_some_and(|limit| request.output_limits.max_tokens > limit)
        || !state
            .config
            .deployment
            .capability
            .output_formats
            .contains(&request.requested_output_format)
    {
        return rejection(
            &request,
            StatusCode::BAD_REQUEST,
            RejectionCode::InvalidRequest,
            "requested input or output limits exceed worker capability",
            None,
        );
    }

    // Serialize the transition from accepting work to draining with runtime
    // admission. A holder either observes draining or transfers authoritative
    // ownership before `begin_draining` can return.
    let admission_guard = Arc::clone(&state.admission_gate).read_owned().await;
    if state.draining.load(Ordering::Acquire) {
        return rejection(
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::WorkerDraining,
            "worker is draining",
            None,
        );
    }

    match state.reserve_attempt(&request) {
        ReserveAttempt::Reserved => {}
        ReserveAttempt::AlreadyOwned => {
            return (
                StatusCode::CONFLICT,
                "attempt is already owned; acceptance is unknown, query or cancel it",
            )
                .into_response()
        }
        ReserveAttempt::Conflict => {
            return rejection(
                &request,
                StatusCode::CONFLICT,
                RejectionCode::DuplicateAttemptConflict,
                "attempt identity was reused with different content",
                None,
            )
        }
        ReserveAttempt::Full => {
            return rejection(
                &request,
                StatusCode::TOO_MANY_REQUESTS,
                RejectionCode::CapacityExhausted,
                "attempt retention is exhausted",
                None,
            )
        }
    }

    let permit = match Arc::clone(&state.capacity).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            state.remove_reservation(&request.attempt_id);
            return rejection(
                &request,
                StatusCode::TOO_MANY_REQUESTS,
                RejectionCode::CapacityExhausted,
                "worker capacity is exhausted",
                Some(25),
            );
        }
    };

    let remaining_before_admission =
        Duration::from_millis(request.remaining_time_ms).saturating_sub(received_at.elapsed());
    let admission_budget =
        remaining_before_admission.min(Duration::from_millis(request.max_queue_wait_ms));
    let admission_slot = Arc::new(Mutex::new(AdmissionSlot::default()));
    let admission_ready = Arc::new(tokio::sync::Notify::new());
    let admission_wait = admission_ready.notified();
    tokio::pin!(admission_wait);
    let admission_state = Arc::clone(&state);
    let admission_request = request.clone();
    let slot_for_admission = Arc::clone(&admission_slot);
    let ready_for_admission = Arc::clone(&admission_ready);
    tokio::spawn(async move {
        let result = admission_state.executor.admit(&admission_request).await;
        let handoff = AdmissionHandoff {
            result,
            permit,
            admission_guard,
        };
        let abandoned_handoff = {
            let mut slot = slot_for_admission
                .lock()
                .expect("worker admission slot poisoned");
            if slot.abandoned {
                Some(handoff)
            } else {
                slot.handoff = Some(handoff);
                None
            }
        };
        ready_for_admission.notify_one();
        if let Some(handoff) = abandoned_handoff {
            settle_abandoned_admission(admission_state, admission_request, handoff).await;
        }
    });
    let handoff = match tokio::time::timeout(admission_budget, &mut admission_wait).await {
        Ok(()) => admission_slot
            .lock()
            .expect("worker admission slot poisoned")
            .handoff
            .take()
            .expect("admission notification carries ownership"),
        Err(_) => {
            // Resolve the timeout/completion race while holding the slot. If
            // admission already published ownership, it wins. Otherwise mark
            // the receiver abandoned before returning the uncertain response.
            let completed = {
                let mut slot = admission_slot
                    .lock()
                    .expect("worker admission slot poisoned");
                if let Some(handoff) = slot.handoff.take() {
                    Some(handoff)
                } else {
                    slot.abandoned = true;
                    None
                }
            };
            if let Some(handoff) = completed {
                handoff
            } else {
                // The runtime may already own an exact Engine session. Continue
                // admission under the permit; the detached path cancels and fences
                // any later success. This is intentionally not a typed rejection.
                state.mark_cancellation_requested(&request.attempt_id);
                return (
                    StatusCode::GATEWAY_TIMEOUT,
                    "runtime admission outcome is unknown; query or cancel this attempt",
                )
                    .into_response();
            }
        }
    };
    let admitted = match handoff.result {
        Ok(admitted) => admitted,
        Err(failure) => {
            drop(handoff.admission_guard);
            drop(handoff.permit);
            state.remove_reservation(&request.attempt_id);
            return rejection(
                &request,
                rejection_status(failure.code),
                failure.code,
                bounded_message(failure.message),
                failure.retry_after_ms,
            );
        }
    };
    let permit = handoff.permit;
    let admission_guard = handoff.admission_guard;
    let execution = admitted.into_execution();
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let cancel_was_requested = state.install_admission(&request.attempt_id, cancel_tx.clone());
    if cancel_was_requested {
        let _ = cancel_tx.send(true);
    }

    let (tx, mut rx) = mpsc::channel::<Bytes>(state.config.event_channel_capacity);
    let remaining_time =
        Duration::from_millis(request.remaining_time_ms).saturating_sub(received_at.elapsed());
    tokio::spawn(run_invocation(
        Arc::clone(&state),
        request,
        permit,
        execution,
        cancel_rx,
        remaining_time,
        tx,
    ));
    drop(admission_guard);
    let output = stream! {
        while let Some(bytes) = rx.recv().await {
            yield Ok::<Bytes, Infallible>(bytes);
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
        .body(Body::from_stream(output))
        .expect("static worker response")
}

async fn run_invocation<E: InvocationExecutor>(
    state: Arc<WorkerState<E>>,
    request: InvocationRequest,
    permit: OwnedSemaphorePermit,
    mut execution: Box<dyn AdmittedExecution>,
    mut cancel: watch::Receiver<bool>,
    remaining_time: Duration,
    tx: mpsc::Sender<Bytes>,
) {
    let accepted = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence: 0,
        event: InvocationEventKind::Accepted {
            worker_id: state.config.descriptor.worker_id.clone(),
            node_id: state.config.descriptor.node_id.clone(),
            incarnation_id: state.config.descriptor.incarnation_id.clone(),
            deployment_id: state.config.deployment.deployment_id.clone(),
            model_generation: state.config.deployment.model_generation,
        },
    };
    let accepted_delivered = matches!(
        try_send_event(&state.config, &tx, accepted),
        EventSendResult::Sent
    );
    state.update_attempt(&request.attempt_id, AttemptState::Running, Some(0));

    let deadline = tokio::time::sleep(remaining_time);
    tokio::pin!(deadline);
    let mut sequence = 1;
    let mut output_bytes = 0usize;
    let mut timed_out = false;
    let mut cancellation_requested = !accepted_delivered || *cancel.borrow();
    let mut terminal = None;
    if cancellation_requested {
        execution.request_cancel();
        state.update_attempt(&request.attempt_id, AttemptState::ExecutionStopping, None);
    }

    while terminal.is_none() && !cancellation_requested {
        tokio::select! {
            biased;
            () = &mut deadline, if !timed_out => {
                timed_out = true;
                cancellation_requested = true;
                execution.request_cancel();
                state.update_attempt(
                    &request.attempt_id,
                    AttemptState::ExecutionStopping,
                    None,
                );
            }
            () = tx.closed(), if !cancellation_requested => {
                cancellation_requested = true;
                execution.request_cancel();
                state.update_attempt(
                    &request.attempt_id,
                    AttemptState::ExecutionStopping,
                    None,
                );
            }
            changed = cancel.changed(), if !cancellation_requested => {
                if changed.is_ok() && *cancel.borrow() {
                    cancellation_requested = true;
                    execution.request_cancel();
                    state.update_attempt(
                        &request.attempt_id,
                        AttemptState::ExecutionStopping,
                        None,
                    );
                }
            }
            event = execution.next_event() => {
                match event {
                    Some(ExecutionEvent::TextDelta(text)) if !cancellation_requested => {
                        output_bytes = output_bytes.saturating_add(text.len());
                        if output_bytes as u64 > request.output_limits.max_bytes {
                            terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                                code: InvocationErrorCode::OutputLimitExceeded,
                                message: "runtime output exceeded the requested byte limit".into(),
                            }));
                            cancellation_requested = true;
                            execution.request_cancel();
                            state.update_attempt(
                                &request.attempt_id,
                                AttemptState::ExecutionStopping,
                                None,
                            );
                            continue;
                        }
                        let event = InvocationEvent {
                            schema_version: PROTOCOL_V1,
                            request_id: request.request_id.clone(),
                            attempt_id: request.attempt_id.clone(),
                            sequence,
                            event: InvocationEventKind::TextDelta { text },
                        };
                        match try_send_event(&state.config, &tx, event) {
                            EventSendResult::Sent => {
                                state.update_attempt(
                                    &request.attempt_id,
                                    AttemptState::Running,
                                    Some(sequence),
                                );
                                sequence = sequence.saturating_add(1);
                            }
                            EventSendResult::Oversized => {
                                terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                                    code: InvocationErrorCode::OutputLimitExceeded,
                                    message: "encoded output event exceeded the worker limit".into(),
                                }));
                                cancellation_requested = true;
                                execution.request_cancel();
                            }
                            EventSendResult::Unavailable => {
                                cancellation_requested = true;
                                execution.request_cancel();
                            }
                        }
                    }
                    Some(ExecutionEvent::TextDelta(_)) => {}
                    Some(ExecutionEvent::Completed {
                        text,
                        finish_reason,
                        input_tokens,
                        output_tokens,
                    }) => {
                        if let Some(text) = text.filter(|text| !text.is_empty()) {
                            output_bytes = output_bytes.saturating_add(text.len());
                            if output_bytes as u64 > request.output_limits.max_bytes {
                                terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                                    code: InvocationErrorCode::OutputLimitExceeded,
                                    message: "runtime output exceeded the requested byte limit".into(),
                                }));
                                execution.request_cancel();
                                continue;
                            }
                            let delta = InvocationEvent {
                                schema_version: PROTOCOL_V1,
                                request_id: request.request_id.clone(),
                                attempt_id: request.attempt_id.clone(),
                                sequence,
                                event: InvocationEventKind::TextDelta { text },
                            };
                            match try_send_event(&state.config, &tx, delta) {
                                EventSendResult::Sent => {
                                    state.update_attempt(
                                        &request.attempt_id,
                                        AttemptState::Running,
                                        Some(sequence),
                                    );
                                    sequence = sequence.saturating_add(1);
                                }
                                EventSendResult::Oversized => {
                                    terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                                        code: InvocationErrorCode::OutputLimitExceeded,
                                        message: "encoded output event exceeded the worker limit".into(),
                                    }));
                                    execution.request_cancel();
                                    continue;
                                }
                                EventSendResult::Unavailable => {
                                    cancellation_requested = true;
                                    execution.request_cancel();
                                }
                            }
                        }
                        terminal = Some(TerminalEvent::Completed {
                            finish_reason,
                            usage: Usage {
                                input_tokens,
                                output_tokens,
                            },
                        });
                    }
                    Some(ExecutionEvent::Failed(failure)) => {
                        terminal = Some(TerminalEvent::Failed(failure));
                    }
                    None => break,
                }
            }
        }
    }

    // A terminal runtime event and a deadline are not teardown proof. Retain
    // worker capacity until the admitted runtime owner confirms cleanup.
    let teardown = execution.wait_for_teardown().await;
    if teardown == ExecutionTeardown::Unconfirmed {
        state.update_attempt(&request.attempt_id, AttemptState::ExecutionStopping, None);
        // Fail closed: losing capacity is safer than advertising a credit
        // while native work may still exist in this incarnation.
        std::mem::forget(permit);
        return;
    }
    let terminal = if timed_out {
        TerminalEvent::Failed(ExecutionFailure {
            code: InvocationErrorCode::DeadlineExceeded,
            message: "invocation deadline elapsed; execution teardown is confirmed".into(),
        })
    } else if cancellation_requested || teardown == ExecutionTeardown::Cancelled {
        TerminalEvent::Cancelled
    } else if teardown == ExecutionTeardown::Failed {
        match terminal {
            Some(TerminalEvent::Failed(failure)) => TerminalEvent::Failed(failure),
            _ => TerminalEvent::Failed(ExecutionFailure {
                code: InvocationErrorCode::ExecutionFailed,
                message: "runtime execution ended without successful teardown".into(),
            }),
        }
    } else {
        terminal.unwrap_or_else(|| {
            TerminalEvent::Failed(ExecutionFailure {
                code: InvocationErrorCode::Internal,
                message: "runtime execution ended without a terminal event".into(),
            })
        })
    };
    publish_terminal(&state, &request, &tx, sequence, terminal);
}

async fn settle_abandoned_admission<E: InvocationExecutor>(
    state: Arc<WorkerState<E>>,
    request: InvocationRequest,
    handoff: AdmissionHandoff,
) {
    let AdmissionHandoff {
        result,
        permit,
        admission_guard,
    } = handoff;
    let Ok(admitted) = result else {
        state.remove_reservation(&request.attempt_id);
        drop(admission_guard);
        return;
    };
    let execution = admitted.into_execution();
    execution.request_cancel();
    state.update_attempt(&request.attempt_id, AttemptState::ExecutionStopping, None);
    drop(admission_guard);
    match execution.wait_for_teardown().await {
        ExecutionTeardown::Completed => {
            state.update_attempt(&request.attempt_id, AttemptState::Completed, None)
        }
        ExecutionTeardown::Cancelled => {
            state.update_attempt(&request.attempt_id, AttemptState::Cancelled, None)
        }
        ExecutionTeardown::Failed => {
            state.update_attempt(&request.attempt_id, AttemptState::Failed, None)
        }
        ExecutionTeardown::Unconfirmed => {
            std::mem::forget(permit);
            return;
        }
    }
    drop(permit);
}

enum TerminalEvent {
    Completed {
        finish_reason: FinishReason,
        usage: Usage,
    },
    Cancelled,
    Failed(ExecutionFailure),
}

fn publish_terminal<E>(
    state: &WorkerState<E>,
    request: &InvocationRequest,
    tx: &mpsc::Sender<Bytes>,
    sequence: u64,
    terminal: TerminalEvent,
) {
    let (attempt_state, event) = match terminal {
        TerminalEvent::Completed {
            finish_reason,
            usage,
        } => (
            AttemptState::Completed,
            InvocationEventKind::Completed {
                finish_reason,
                usage: Some(usage),
            },
        ),
        TerminalEvent::Cancelled => (
            AttemptState::Cancelled,
            InvocationEventKind::Cancelled {
                reason: Some("requested".into()),
            },
        ),
        TerminalEvent::Failed(failure) => (
            AttemptState::Failed,
            InvocationEventKind::Error {
                code: failure.code,
                message: bounded_message(failure.message),
            },
        ),
    };
    let event = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence,
        event,
    };
    let published = matches!(
        try_send_event(&state.config, tx, event),
        EventSendResult::Sent
    );
    state.update_attempt(
        &request.attempt_id,
        attempt_state,
        published.then_some(sequence),
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventSendResult {
    Sent,
    Oversized,
    Unavailable,
}

fn try_send_event(
    config: &WorkerConfig,
    tx: &mpsc::Sender<Bytes>,
    event: InvocationEvent,
) -> EventSendResult {
    let Ok(mut encoded) = serde_json::to_vec(&event) else {
        return EventSendResult::Oversized;
    };
    if encoded.len().saturating_add(1) > config.max_event_bytes {
        return EventSendResult::Oversized;
    }
    encoded.push(b'\n');
    match tx.try_send(Bytes::from(encoded)) {
        Ok(()) => EventSendResult::Sent,
        Err(_) => EventSendResult::Unavailable,
    }
}

async fn query_attempt<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    Path(raw_attempt_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(attempt_id) = AttemptId::new(raw_attempt_id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let mut table = state
        .attempts
        .lock()
        .expect("worker attempt table poisoned");
    let Some(record) = table.records.get(&attempt_id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if WorkerState::<E>::record_is_expired(record, Instant::now()) {
        let identity = record.identity.clone();
        WorkerState::<E>::remove_record(&mut table, &attempt_id);
        return (
            StatusCode::GONE,
            Json(AttemptQueryResponse {
                schema_version: PROTOCOL_V1,
                worker_id: state.config.descriptor.worker_id.clone(),
                identity,
                state: AttemptState::Expired,
                last_sequence: None,
            }),
        )
            .into_response();
    }
    Json(AttemptQueryResponse {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.descriptor.worker_id.clone(),
        identity: record.identity.clone(),
        state: record.state,
        last_sequence: record.last_sequence,
    })
    .into_response()
}

async fn cancel_attempt<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    Path(attempt_id): Path<String>,
    headers: HeaderMap,
    Json(request): Json<CancelAttemptRequest>,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(path_attempt_id) = AttemptId::new(attempt_id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if request.schema_version.major != PROTOCOL_V1.major
        || path_attempt_id != request.identity.attempt_id
        || request.identity.incarnation_id != state.config.descriptor.incarnation_id
    {
        return StatusCode::CONFLICT.into_response();
    }
    let (disposition, cancel) = {
        let mut table = state
            .attempts
            .lock()
            .expect("worker attempt table poisoned");
        let expired = table
            .records
            .get(&request.identity.attempt_id)
            .filter(|record| record.identity == request.identity)
            .is_some_and(|record| WorkerState::<E>::record_is_expired(record, Instant::now()));
        if expired {
            WorkerState::<E>::remove_record(&mut table, &request.identity.attempt_id);
            return Json(CancelAttemptResponse {
                schema_version: PROTOCOL_V1,
                worker_id: state.config.descriptor.worker_id.clone(),
                identity: request.identity,
                disposition: CancelDisposition::Expired,
            })
            .into_response();
        }
        let Some(record) = table
            .records
            .get_mut(&request.identity.attempt_id)
            .filter(|record| record.identity == request.identity)
        else {
            drop(table);
            state.insert_cancel_tombstone(request.identity.clone());
            return Json(CancelAttemptResponse {
                schema_version: PROTOCOL_V1,
                worker_id: state.config.descriptor.worker_id.clone(),
                identity: request.identity,
                disposition: CancelDisposition::Unknown,
            })
            .into_response();
        };
        if record.state.is_terminal() {
            (CancelDisposition::AlreadyTerminal, None)
        } else if record.cancel_requested {
            (CancelDisposition::AlreadyRequested, None)
        } else {
            record.cancel_requested = true;
            record.state = AttemptState::CancellationRequested;
            (CancelDisposition::Requested, record.cancel.clone())
        }
    };

    let disposition = if let Some(cancel) = cancel {
        if cancel.send(true).is_ok() {
            state.update_attempt(
                &request.identity.attempt_id,
                AttemptState::ExecutionStopping,
                None,
            );
            disposition
        } else {
            CancelDisposition::Unknown
        }
    } else {
        disposition
    };
    Json(CancelAttemptResponse {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.descriptor.worker_id.clone(),
        identity: request.identity,
        disposition,
    })
    .into_response()
}

fn rejection(
    request: &InvocationRequest,
    status: StatusCode,
    code: RejectionCode,
    message: impl Into<String>,
    retry_after_ms: Option<u64>,
) -> Response {
    let mut rejection = InvocationRejection::new(
        request.request_id.clone(),
        request.attempt_id.clone(),
        code,
        bounded_message(message.into()),
    );
    rejection.retry_after_ms = retry_after_ms;
    (status, Json(rejection)).into_response()
}

fn rejection_status(code: RejectionCode) -> StatusCode {
    match code {
        RejectionCode::Unauthenticated => StatusCode::UNAUTHORIZED,
        RejectionCode::Unauthorized | RejectionCode::PolicyDenied => StatusCode::FORBIDDEN,
        RejectionCode::UnsupportedProtocolVersion
        | RejectionCode::WrongWorkerIncarnation
        | RejectionCode::WrongModelGeneration
        | RejectionCode::DuplicateAttemptConflict => StatusCode::CONFLICT,
        RejectionCode::UnknownDeployment => StatusCode::NOT_FOUND,
        RejectionCode::IncompatibleTask => StatusCode::UNPROCESSABLE_ENTITY,
        RejectionCode::ModelNotReady | RejectionCode::WorkerDraining => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        RejectionCode::CapacityExhausted | RejectionCode::QueueWaitExceeded => {
            StatusCode::TOO_MANY_REQUESTS
        }
        RejectionCode::InvalidRequest => StatusCode::BAD_REQUEST,
    }
}

fn bounded_message(mut message: String) -> String {
    const MAX_MESSAGE_BYTES: usize = 1024;
    if message.len() <= MAX_MESSAGE_BYTES {
        return message;
    }
    let mut end = MAX_MESSAGE_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request};
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn credentials() -> ServiceCredentials {
        ServiceCredentials {
            credential_id: id("worker-credential"),
            bearer_token: ServiceBearerToken::new("worker-secret").unwrap(),
        }
    }

    fn config() -> WorkerConfig {
        WorkerConfig {
            descriptor: WorkerDescriptor {
                schema_version: PROTOCOL_V1,
                supported_protocol_versions: vec![PROTOCOL_V1],
                worker_id: id("cpu-worker"),
                node_id: id("local-node"),
                incarnation_id: id("incarnation-1"),
                build_version: "test".into(),
                assignment: DeviceAssignment::Cpu {
                    thread_budget: 1,
                    affinity: Vec::new(),
                    host_memory_limit_bytes: 64 * 1024 * 1024,
                },
                features: BTreeSet::from([
                    WorkerFeature::Streaming,
                    WorkerFeature::Cancellation,
                    WorkerFeature::AttemptQuery,
                ]),
            },
            deployment: LoadedDeployment {
                deployment_id: id("lfm-cpu-v1"),
                public_model: id("tiny-lfm"),
                artifact_revision: id("tiny-fixture-v1"),
                model_generation: ModelGeneration::new(1).unwrap(),
                task: TaskKind::Chat,
                backend: BackendKind::Cpu,
                precision: "f32".into(),
                execution_representation: "tiny-lfm".into(),
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
                    max_context_tokens: Some(32),
                    max_output_tokens: Some(32),
                },
            },
            credentials: credentials(),
            max_active_invocations: 1,
            max_request_bytes: 4096,
            max_retained_attempts: 4,
            attempt_retention: Duration::from_secs(300),
            event_channel_capacity: 4,
            max_event_bytes: 4096,
        }
    }

    fn invocation() -> InvocationRequest {
        InvocationRequest {
            schema_version: PROTOCOL_V1,
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            expected_worker_incarnation: id("incarnation-1"),
            deployment_id: id("lfm-cpu-v1"),
            expected_model_generation: ModelGeneration::new(1).unwrap(),
            caller: GatewayAttestedCallerContext {
                tenant_id: id("tenant-1"),
                caller_id: id("caller-1"),
                policy_revision: id("policy-1"),
                permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                allowed_data_regions: vec!["local".into()],
            },
            task: TaskKind::Chat,
            service_class: ServiceClass::Interactive,
            remaining_time_ms: 5_000,
            max_queue_wait_ms: 1_000,
            output_limits: OutputLimits {
                max_tokens: 8,
                max_bytes: 1024,
            },
            requested_output_format: OutputFormat::Text,
            session_id: None,
            request_digest: id("sha256:test"),
            input: InvocationInput::Chat {
                input: ChatInput {
                    messages: vec![ChatMessage {
                        role: ChatRole::User,
                        content: "hello".into(),
                    }],
                },
                parameters: ChatParameters::default(),
            },
        }
    }

    fn authorized_request(method: &str, uri: &str, body: Body) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(
                SERVICE_AUTHORIZATION_HEADER,
                format!("{SERVICE_AUTH_SCHEME} worker-secret"),
            )
            .header(SERVICE_CREDENTIAL_ID_HEADER, "worker-credential")
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(body)
            .unwrap()
    }

    struct ScriptExecutor {
        events: Mutex<Option<VecDeque<ExecutionEvent>>>,
        teardown: ExecutionTeardown,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for ScriptExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            let events = self.events.lock().unwrap().take().unwrap();
            Ok(AdmittedInvocation::new(Box::new(ScriptExecution {
                events,
                teardown: self.teardown,
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    struct ScriptExecution {
        events: VecDeque<ExecutionEvent>,
        teardown: ExecutionTeardown,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AdmittedExecution for ScriptExecution {
        async fn next_event(&mut self) -> Option<ExecutionEvent> {
            self.events.pop_front()
        }

        fn request_cancel(&self) {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
        }

        async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown {
            self.teardown
        }
    }

    #[tokio::test]
    async fn accepted_stream_is_ordered_and_terminal_after_teardown() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::from([
                ExecutionEvent::TextDelta("tiny response".into()),
                ExecutionEvent::Completed {
                    text: None,
                    finish_reason: FinishReason::Stop,
                    input_tokens: 2,
                    output_tokens: 2,
                },
            ]))),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let events = body
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<InvocationEvent>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0].event,
            InvocationEventKind::Accepted { .. }
        ));
        assert!(matches!(
            events[1].event,
            InvocationEventKind::TextDelta { .. }
        ));
        assert!(matches!(
            events[2].event,
            InvocationEventKind::Completed { .. }
        ));
        assert_eq!(service.active_invocations(), 0);
    }

    #[tokio::test]
    async fn bounded_output_backpressure_requests_cancel_without_holding_more_events() {
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let mut events = (0..8)
            .map(|index| ExecutionEvent::TextDelta(format!("delta-{index}")))
            .collect::<VecDeque<_>>();
        events.push_back(ExecutionEvent::Completed {
            text: None,
            finish_reason: FinishReason::Stop,
            input_tokens: 1,
            output_tokens: 8,
        });
        let executor = ScriptExecutor {
            events: Mutex::new(Some(events)),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::clone(&cancel_calls),
        };
        let mut worker_config = config();
        worker_config.event_channel_capacity = 3;
        let service = WorkerService::new(worker_config, executor).unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while cancel_calls.load(Ordering::Acquire) == 0 || service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(cancel_calls.load(Ordering::Acquire), 1);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        assert!(body.len() <= 3 * config().max_event_bytes);
    }

    #[tokio::test]
    async fn unconfirmed_teardown_never_publishes_terminal_or_releases_capacity() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Unconfirmed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let request = invocation();
        let attempt_id = request.attempt_id.clone();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let events = body
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<InvocationEvent>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].event,
            InvocationEventKind::Accepted { .. }
        ));
        assert_eq!(service.active_invocations(), 1);

        let query = service
            .router()
            .oneshot(authorized_request(
                "GET",
                &format!("{INVOCATIONS_PATH}/{attempt_id}"),
                Body::empty(),
            ))
            .await
            .unwrap();
        let body = to_bytes(query.into_body(), 4096).await.unwrap();
        let attempt: AttemptQueryResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(attempt.state, AttemptState::ExecutionStopping);
    }

    struct LateAdmissionExecutor {
        release: Arc<tokio::sync::Notify>,
        completed: Arc<AtomicBool>,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for LateAdmissionExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            self.release.notified().await;
            self.completed.store(true, Ordering::Release);
            Ok(AdmittedInvocation::new(Box::new(ScriptExecution {
                events: VecDeque::new(),
                teardown: ExecutionTeardown::Cancelled,
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    #[tokio::test]
    async fn expired_admission_wait_retains_ownership_and_is_not_a_rejection() {
        let release = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(AtomicBool::new(false));
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            LateAdmissionExecutor {
                release: Arc::clone(&release),
                completed: Arc::clone(&completed),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let mut request = invocation();
        request.max_queue_wait_ms = 1;
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert!(serde_json::from_slice::<InvocationRejection>(&body).is_err());
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(service.active_invocations(), 1);

        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(completed.load(Ordering::Acquire));
        assert_eq!(cancel_calls.load(Ordering::Acquire), 1);
    }

    struct DelayedExecutor {
        entered: Arc<tokio::sync::Notify>,
        release_admission: Arc<tokio::sync::Notify>,
        finish_execution: Arc<tokio::sync::Notify>,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for DelayedExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            self.entered.notify_one();
            self.release_admission.notified().await;
            Ok(AdmittedInvocation::new(Box::new(DelayedExecution {
                finish_execution: Arc::clone(&self.finish_execution),
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    struct DelayedExecution {
        finish_execution: Arc<tokio::sync::Notify>,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AdmittedExecution for DelayedExecution {
        async fn next_event(&mut self) -> Option<ExecutionEvent> {
            self.finish_execution.notified().await;
            None
        }

        fn request_cancel(&self) {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
        }

        async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown {
            self.finish_execution.notified().await;
            ExecutionTeardown::Cancelled
        }
    }

    #[tokio::test]
    async fn cancel_before_admission_is_preserved_and_capacity_waits_for_teardown() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let request = invocation();
        let identity = AttemptIdentity::from(&request);
        let invoke_router = service.router();
        let invoke_task = tokio::spawn(async move {
            invoke_router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(serde_json::to_vec(&request).unwrap()),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        let cancel_path = format!("{INVOCATIONS_PATH}/{}/cancel", identity.attempt_id);
        let cancel_response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                &cancel_path,
                Body::from(
                    serde_json::to_vec(&CancelAttemptRequest {
                        schema_version: PROTOCOL_V1,
                        identity,
                    })
                    .unwrap(),
                ),
            ))
            .await
            .unwrap();
        let body = to_bytes(cancel_response.into_body(), 4096).await.unwrap();
        let cancelled: CancelAttemptResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(cancelled.disposition, CancelDisposition::Requested);

        release_admission.notify_one();
        let response = invoke_task.await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        tokio::time::timeout(Duration::from_secs(1), async {
            while cancel_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(service.active_invocations(), 1);

        finish_execution.notify_one();
        let _ = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn request_body_limit_is_enforced_before_json_allocation() {
        let mut worker_config = config();
        worker_config.max_request_bytes = 64;
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(worker_config, executor).unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn attested_caller_without_invoke_permission_is_rejected_before_admission() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let mut request = invocation();
        request.caller.permitted_actions.clear();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let rejection: InvocationRejection = serde_json::from_slice(&body).unwrap();
        assert_eq!(rejection.code, RejectionCode::PolicyDenied);
        assert_eq!(service.active_invocations(), 0);
    }

    #[tokio::test]
    async fn drain_waits_for_in_progress_admission_then_rejects_new_work() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap();
        let first_request = invocation();
        let invoke_router = service.router();
        let invoke_task = tokio::spawn(async move {
            invoke_router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(serde_json::to_vec(&first_request).unwrap()),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();

        let drain_service = service.clone();
        let drain_task = tokio::spawn(async move { drain_service.begin_draining().await });
        tokio::task::yield_now().await;
        assert!(!drain_task.is_finished());

        release_admission.notify_one();
        let first_response = invoke_task.await.unwrap();
        assert_eq!(first_response.status(), StatusCode::OK);
        tokio::time::timeout(Duration::from_secs(1), drain_task)
            .await
            .unwrap()
            .unwrap();

        let mut second_request = invocation();
        second_request.request_id = id("request-2");
        second_request.attempt_id = id("attempt-2");
        second_request.request_digest = id("sha256:test-2");
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&second_request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let rejection: InvocationRejection = serde_json::from_slice(&body).unwrap();
        assert_eq!(rejection.code, RejectionCode::WorkerDraining);

        drop(first_response);
        finish_execution.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn duplicate_owned_attempt_is_an_uncertain_response_not_a_rejection() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap();
        let request = invocation();
        let first_router = service.router();
        let first_body = serde_json::to_vec(&request).unwrap();
        let first_task = tokio::spawn(async move {
            first_router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(first_body),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        release_admission.notify_one();
        let first_response = first_task.await.unwrap();

        let duplicate = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(duplicate.status(), StatusCode::CONFLICT);
        let body = to_bytes(duplicate.into_body(), 4096).await.unwrap();
        assert!(serde_json::from_slice::<InvocationRejection>(&body).is_err());

        drop(first_response);
        finish_execution.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn dropping_http_stream_requests_cancel_and_holds_capacity_until_teardown() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let router = service.router();
        let task = tokio::spawn(async move {
            router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(serde_json::to_vec(&invocation()).unwrap()),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        release_admission.notify_one();
        let response = task.await.unwrap();
        assert_eq!(service.active_invocations(), 1);
        drop(response);

        tokio::time::timeout(Duration::from_secs(1), async {
            while cancel_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(service.active_invocations(), 1);

        finish_execution.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_tombstone_expires_after_the_declared_retention_window() {
        let mut worker_config = config();
        worker_config.attempt_retention = Duration::from_secs(1);
        let service = WorkerService::new(
            worker_config,
            ScriptExecutor {
                events: Mutex::new(Some(VecDeque::new())),
                teardown: ExecutionTeardown::Completed,
                cancel_calls: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap();
        let request = invocation();
        service
            .state
            .insert_cancel_tombstone(AttemptIdentity::from(&request));
        assert_eq!(
            service.state.reserve_attempt(&request),
            ReserveAttempt::Conflict
        );

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(
            service.state.reserve_attempt(&request),
            ReserveAttempt::Reserved
        );
    }
}

use crate::{
    ArtifactRevision, AttemptId, CallerId, CredentialId, DeploymentId, DeviceId, IncarnationId,
    ModelAlias, ModelGeneration, NodeId, PolicyRevision, RequestDigest, RequestId, ServiceId,
    SessionId, TenantId, WorkerId, PROTOCOL_V1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MAX_CHAT_MESSAGES: usize = 256;
pub const MAX_CHAT_MESSAGE_BYTES: usize = 256 * 1024;
pub const MAX_STOP_SEQUENCES: usize = 16;
pub const MAX_STOP_SEQUENCE_BYTES: usize = 1024;
pub const MAX_CALLER_REGIONS: usize = 32;
pub const MAX_REMAINING_TIME_MS: u64 = 86_400_000; // 24 hours

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
}

impl SchemaVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Returns true when `peer` uses this version's major and implements at least this minor.
    pub const fn is_supported_by(self, peer: Self) -> bool {
        self.major == peer.major && self.minor <= peer.minor
    }

    /// Additive-minor tolerance: peers sharing this major version always
    /// interoperate. Older peers omit optional fields (decoded as absent);
    /// newer peers only ever add fields this version already tolerates.
    pub const fn shares_major_with(self, peer: Self) -> bool {
        self.major == peer.major
    }
}

impl Default for SchemaVersion {
    fn default() -> Self {
        PROTOCOL_V1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
}

/// Explicit process resource assignment. The supervisor, not a request, selects this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum DeviceAssignment {
    Cpu {
        thread_budget: u16,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        affinity: Vec<u16>,
        host_memory_limit_bytes: u64,
    },
    Metal {
        device_id: DeviceId,
        process_local_device_index: u32,
        shared_memory_limit_bytes: u64,
    },
    Cuda {
        device_uuid: DeviceId,
        process_local_device_index: u32,
        device_memory_limit_bytes: u64,
        host_memory_limit_bytes: u64,
    },
}

impl DeviceAssignment {
    pub const fn backend(&self) -> BackendKind {
        match self {
            Self::Cpu { .. } => BackendKind::Cpu,
            Self::Metal { .. } => BackendKind::Metal,
            Self::Cuda { .. } => BackendKind::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Chat,
    TextToSpeech,
    SpeechToText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceClass {
    Realtime,
    Interactive,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputFormat {
    ChatMessages,
    PcmAudio,
    EncodedAudio,
    ArtifactReference,
    /// Bounded plain text input (realtime TTS-stream stage, protocol minor 2).
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Text,
    Json,
    PcmAudio,
    EncodedAudio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationBehavior {
    Cooperative,
    TeardownRequired,
    NotSupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub task: TaskKind,
    pub streaming: bool,
    pub realtime: bool,
    pub cancellation: CancellationBehavior,
    pub accepted_input_formats: BTreeSet<InputFormat>,
    pub output_formats: BTreeSet<OutputFormat>,
    pub max_input_bytes: u64,
    pub max_context_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelReadiness {
    Loading,
    Warming,
    Ready,
    Draining,
    Failed,
}

/// One model deployment loaded on a worker.
///
/// The routing-signal fields are additive (protocol minor 1): workers set them
/// only when the engine reports them, and receivers must treat absence as
/// "signal unavailable" rather than as a zero observation. Counters are
/// process-cumulative for the deployment's engine; rates are receiver-facing
/// observations, not benchmarks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadedDeployment {
    pub deployment_id: DeploymentId,
    pub public_model: ModelAlias,
    pub artifact_revision: ArtifactRevision,
    pub model_generation: ModelGeneration,
    pub task: TaskKind,
    pub backend: BackendKind,
    pub precision: String,
    pub execution_representation: String,
    pub tokenizer_revision: Option<ArtifactRevision>,
    pub readiness: ModelReadiness,
    pub capability: Capability,
    /// Managed-KV arena utilization, percent 0-100.
    pub kv_cache_usage_pct: Option<f64>,
    /// Cumulative managed-KV shared-prefix hits.
    pub prefix_hits_total: Option<u64>,
    /// Cumulative shared-prefix lookups (hits plus misses).
    pub prefix_queries_total: Option<u64>,
    /// Cumulative shared-prefix evictions under arena pressure.
    pub prefix_evictions_total: Option<u64>,
    /// Host-resident offloaded KV pages currently held by the DS4 host pool.
    pub kv_host_pages: Option<u64>,
    /// Cumulative pages demoted from the device arena to the host pool (DS4).
    pub kv_demotions_total: Option<u64>,
    /// Cumulative host pages promoted back into the device arena (DS4).
    pub kv_promotions_total: Option<u64>,
    /// Mean promotion copy latency in seconds across promoted pages (DS4).
    pub kv_promotion_latency_avg_seconds: Option<f64>,
    /// Exponentially weighted output tokens per second over completed
    /// invocations, as observed by the worker.
    pub tokens_out_per_s_ema: Option<f64>,
    /// Admission-credit units one observation (invocation) of this deployment
    /// is priced at by the worker's own permit model.
    pub observation_cost_units: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerFeature {
    Streaming,
    Cancellation,
    AttemptQuery,
    /// Worker serves realtime sessions on `GET /internal/v1/realtime` via the
    /// `izwi-realtime-v1` WebSocket subprotocol (protocol minor 2).
    RealtimeSocket,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerDescriptor {
    pub schema_version: SchemaVersion,
    pub supported_protocol_versions: Vec<SchemaVersion>,
    pub worker_id: WorkerId,
    pub node_id: NodeId,
    pub incarnation_id: IncarnationId,
    pub build_version: String,
    pub assignment: DeviceAssignment,
    pub features: BTreeSet<WorkerFeature>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProcessState {
    Starting,
    Running,
    Draining,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacitySnapshot {
    pub max_active_invocations: u32,
    pub active_invocations: u32,
    pub max_queued_invocations: u32,
    pub queued_invocations: u32,
    pub max_sessions: u32,
    pub reserved_sessions: u32,
    pub available_admission_credits: u32,
    pub outstanding_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub schema_version: SchemaVersion,
    pub worker_id: WorkerId,
    pub node_id: NodeId,
    pub incarnation_id: IncarnationId,
    /// Monotonic only within `incarnation_id`; freshness is measured by the receiver's clock.
    pub status_sequence: u64,
    pub process_state: WorkerProcessState,
    pub deployments: Vec<LoadedDeployment>,
    pub capacity: CapacitySnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermittedAction {
    Invoke,
    CancelOwnInvocation,
    QueryOwnInvocation,
}

/// Caller identity asserted by an authenticated gateway, never copied from public identity headers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayAttestedCallerContext {
    pub tenant_id: TenantId,
    pub caller_id: CallerId,
    pub policy_revision: PolicyRevision,
    pub permitted_actions: BTreeSet<PermittedAction>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_data_regions: Vec<String>,
}

/// Result of authenticating the service-level HTTP credential. This is local handler context and
/// intentionally contains no bearer secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceAuthContext {
    pub service_id: ServiceId,
    pub credential_id: CredentialId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatInput {
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatParameters {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputLimits {
    pub max_tokens: u32,
    pub max_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InvocationInput {
    Chat {
        input: ChatInput,
        parameters: ChatParameters,
    },
}

impl InvocationInput {
    pub const fn task(&self) -> TaskKind {
        match self {
            Self::Chat { .. } => TaskKind::Chat,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvocationRequest {
    pub schema_version: SchemaVersion,
    pub request_id: RequestId,
    pub attempt_id: AttemptId,
    pub expected_worker_incarnation: IncarnationId,
    pub deployment_id: DeploymentId,
    pub expected_model_generation: ModelGeneration,
    pub caller: GatewayAttestedCallerContext,
    pub task: TaskKind,
    pub service_class: ServiceClass,
    /// Remaining end-to-end time budget when sent by the gateway.
    pub remaining_time_ms: u64,
    pub max_queue_wait_ms: u64,
    pub output_limits: OutputLimits,
    pub requested_output_format: OutputFormat,
    pub session_id: Option<SessionId>,
    pub request_digest: RequestDigest,
    pub input: InvocationInput,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ContractValidationError {
    #[error("unsupported protocol version {actual_major}.{actual_minor}; expected major {expected_major}")]
    UnsupportedVersion {
        actual_major: u16,
        actual_minor: u16,
        expected_major: u16,
    },
    #[error("task {declared:?} does not match typed input task {actual:?}")]
    TaskMismatch {
        declared: TaskKind,
        actual: TaskKind,
    },
    #[error("chat must contain between 1 and {MAX_CHAT_MESSAGES} messages")]
    InvalidMessageCount,
    #[error("chat message {index} is empty or exceeds {MAX_CHAT_MESSAGE_BYTES} bytes")]
    InvalidMessageLength { index: usize },
    #[error("chat parameters contain invalid temperature or top_p")]
    InvalidSamplingParameter,
    #[error("chat parameters contain too many or oversized stop sequences")]
    InvalidStopSequences,
    #[error("output limits must be non-zero")]
    InvalidOutputLimits,
    #[error("remaining execution budget must be non-zero")]
    InvalidExecutionBudget,
    #[error("caller context contains too many region constraints")]
    TooManyCallerRegions,
}

impl InvocationRequest {
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        if self.schema_version.major != PROTOCOL_V1.major {
            return Err(ContractValidationError::UnsupportedVersion {
                actual_major: self.schema_version.major,
                actual_minor: self.schema_version.minor,
                expected_major: PROTOCOL_V1.major,
            });
        }
        if self.task != self.input.task() {
            return Err(ContractValidationError::TaskMismatch {
                declared: self.task,
                actual: self.input.task(),
            });
        }
        if self.remaining_time_ms == 0 || self.remaining_time_ms > MAX_REMAINING_TIME_MS {
            return Err(ContractValidationError::InvalidExecutionBudget);
        }
        if self.output_limits.max_tokens == 0 || self.output_limits.max_bytes == 0 {
            return Err(ContractValidationError::InvalidOutputLimits);
        }
        if self.caller.allowed_data_regions.len() > MAX_CALLER_REGIONS {
            return Err(ContractValidationError::TooManyCallerRegions);
        }
        match &self.input {
            InvocationInput::Chat { input, parameters } => {
                if input.messages.is_empty() || input.messages.len() > MAX_CHAT_MESSAGES {
                    return Err(ContractValidationError::InvalidMessageCount);
                }
                for (index, message) in input.messages.iter().enumerate() {
                    if message.content.is_empty() || message.content.len() > MAX_CHAT_MESSAGE_BYTES
                    {
                        return Err(ContractValidationError::InvalidMessageLength { index });
                    }
                }
                if parameters
                    .temperature
                    .is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
                    || parameters
                        .top_p
                        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
                {
                    return Err(ContractValidationError::InvalidSamplingParameter);
                }
                if parameters.stop.len() > MAX_STOP_SEQUENCES
                    || parameters
                        .stop
                        .iter()
                        .any(|stop| stop.is_empty() || stop.len() > MAX_STOP_SEQUENCE_BYTES)
                {
                    return Err(ContractValidationError::InvalidStopSequences);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionCode {
    Unauthenticated,
    Unauthorized,
    UnsupportedProtocolVersion,
    WrongWorkerIncarnation,
    UnknownDeployment,
    WrongModelGeneration,
    IncompatibleTask,
    ModelNotReady,
    WorkerDraining,
    CapacityExhausted,
    QueueWaitExceeded,
    InvalidRequest,
    DuplicateAttemptConflict,
    PolicyDenied,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvocationRejection {
    pub schema_version: SchemaVersion,
    /// Always false for a response constructed with [`InvocationRejection::new`].
    pub accepted: bool,
    pub request_id: RequestId,
    pub attempt_id: AttemptId,
    pub code: RejectionCode,
    pub message: String,
    pub retry_after_ms: Option<u64>,
}

impl InvocationRejection {
    pub fn new(
        request_id: RequestId,
        attempt_id: AttemptId,
        code: RejectionCode,
        message: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: PROTOCOL_V1,
            accepted: false,
            request_id,
            attempt_id,
            code,
            message: message.into(),
            retry_after_ms: None,
        }
    }

    pub const fn is_valid(&self) -> bool {
        !self.accepted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvocationErrorCode {
    InvalidInput,
    ExecutionFailed,
    DeadlineExceeded,
    OutputLimitExceeded,
    WorkerUnavailable,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

/// Bounded per-invocation token accounting.
///
/// `cached_tokens` (minor 3) reports how many input tokens were served from
/// the worker's managed prefix cache. It is always a subset of
/// `input_tokens`; absent means the worker did not measure prefix reuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

/// DS9.3 (minor 3): one top alternative in a token's logprob entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopTokenLogprob {
    pub token: String,
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Vec<u8>,
}

/// DS9.3 (minor 3): per-token logprob entry for one sampled output token.
/// Logprobs are computed from the raw model distribution (log_softmax of the
/// raw logits, before penalties and temperature).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopTokenLogprob>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InvocationEventKind {
    Accepted {
        worker_id: WorkerId,
        node_id: NodeId,
        incarnation_id: IncarnationId,
        deployment_id: DeploymentId,
        model_generation: ModelGeneration,
    },
    TextDelta {
        text: String,
        /// DS9.3: logprob entries for the tokens this delta carries. Absent
        /// when the worker did not collect logprobs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<TokenLogprob>>,
    },
    Usage {
        usage: Usage,
    },
    Completed {
        finish_reason: FinishReason,
        usage: Option<Usage>,
    },
    Error {
        code: InvocationErrorCode,
        message: String,
    },
    Cancelled {
        reason: Option<String>,
    },
}

impl InvocationEventKind {
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Error { .. } | Self::Cancelled { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvocationEvent {
    pub schema_version: SchemaVersion,
    pub request_id: RequestId,
    pub attempt_id: AttemptId,
    pub sequence: u64,
    #[serde(flatten)]
    pub event: InvocationEventKind,
}

impl InvocationEvent {
    pub const fn is_terminal(&self) -> bool {
        self.event.is_terminal()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptState {
    Admitted,
    Queued,
    Running,
    CancellationRequested,
    ExecutionStopping,
    Completed,
    Failed,
    Cancelled,
    Unknown,
    Expired,
}

impl AttemptState {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Unknown and expired records deliberately do not prove that execution never occurred.
    pub const fn proves_execution_stopped(self) -> bool {
        self.is_terminal()
    }
}

/// Exact invocation owner used to fence attempt lookup and cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptIdentity {
    pub request_id: RequestId,
    pub attempt_id: AttemptId,
    pub tenant_id: TenantId,
    pub caller_id: CallerId,
    pub incarnation_id: IncarnationId,
    pub deployment_id: DeploymentId,
    pub model_generation: ModelGeneration,
}

impl From<&InvocationRequest> for AttemptIdentity {
    fn from(request: &InvocationRequest) -> Self {
        Self {
            request_id: request.request_id.clone(),
            attempt_id: request.attempt_id.clone(),
            tenant_id: request.caller.tenant_id.clone(),
            caller_id: request.caller.caller_id.clone(),
            incarnation_id: request.expected_worker_incarnation.clone(),
            deployment_id: request.deployment_id.clone(),
            model_generation: request.expected_model_generation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptQueryResponse {
    pub schema_version: SchemaVersion,
    pub worker_id: WorkerId,
    pub identity: AttemptIdentity,
    pub state: AttemptState,
    pub last_sequence: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelDisposition {
    Requested,
    AlreadyRequested,
    Stopped,
    AlreadyTerminal,
    Unknown,
    Expired,
}

impl CancelDisposition {
    pub const fn confirms_execution_stopped(self) -> bool {
        matches!(self, Self::Stopped | Self::AlreadyTerminal)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelAttemptRequest {
    pub schema_version: SchemaVersion,
    pub identity: AttemptIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelAttemptResponse {
    pub schema_version: SchemaVersion,
    pub worker_id: WorkerId,
    pub identity: AttemptIdentity,
    pub disposition: CancelDisposition,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROTOCOL_MINOR_VERSION;
    use serde_json::json;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn request_fixture() -> InvocationRequest {
        InvocationRequest {
            schema_version: PROTOCOL_V1,
            request_id: id("req-1"),
            attempt_id: id("attempt-1"),
            expected_worker_incarnation: id("inc-1"),
            deployment_id: id("chat-deployment-v1"),
            expected_model_generation: ModelGeneration::new(3).unwrap(),
            caller: GatewayAttestedCallerContext {
                tenant_id: id("tenant-1"),
                caller_id: id("caller-1"),
                policy_revision: id("policy-7"),
                permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                allowed_data_regions: vec!["local".into()],
            },
            task: TaskKind::Chat,
            service_class: ServiceClass::Interactive,
            remaining_time_ms: 10_000,
            max_queue_wait_ms: 250,
            output_limits: OutputLimits {
                max_tokens: 32,
                max_bytes: 4096,
            },
            requested_output_format: OutputFormat::Text,
            session_id: None,
            request_digest: id("sha256:abc123"),
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

    #[test]
    fn current_version_is_compatible_with_additive_minor_only() {
        let next_minor = SchemaVersion::new(1, PROTOCOL_MINOR_VERSION + 1);
        assert!(PROTOCOL_V1.is_supported_by(PROTOCOL_V1));
        assert!(PROTOCOL_V1.is_supported_by(next_minor));
        assert!(SchemaVersion::new(1, 0).is_supported_by(PROTOCOL_V1));
        assert!(!next_minor.is_supported_by(PROTOCOL_V1));
        assert!(!PROTOCOL_V1.is_supported_by(SchemaVersion::new(2, 0)));
        // Additive-minor tolerance: within one major, any minor interoperates.
        assert!(SchemaVersion::new(1, 0).shares_major_with(PROTOCOL_V1));
        assert!(PROTOCOL_V1.shares_major_with(SchemaVersion::new(1, 0)));
        assert!(!PROTOCOL_V1.shares_major_with(SchemaVersion::new(2, 0)));
    }

    fn status_fixture() -> WorkerStatus {
        WorkerStatus {
            schema_version: PROTOCOL_V1,
            worker_id: id("worker-1"),
            node_id: id("node-1"),
            incarnation_id: id("inc-1"),
            status_sequence: 7,
            process_state: WorkerProcessState::Running,
            deployments: vec![LoadedDeployment {
                deployment_id: id("chat-deployment-v1"),
                public_model: id("model-a"),
                artifact_revision: id("artifact-1"),
                model_generation: ModelGeneration::new(3).unwrap(),
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
                    max_input_bytes: 1024,
                    max_context_tokens: None,
                    max_output_tokens: Some(32),
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
            }],
            capacity: CapacitySnapshot {
                max_active_invocations: 4,
                active_invocations: 1,
                max_queued_invocations: 0,
                queued_invocations: 0,
                max_sessions: 0,
                reserved_sessions: 0,
                available_admission_credits: 3,
                outstanding_cost_units: 1,
            },
        }
    }

    #[test]
    fn status_without_routing_signals_decodes_as_unavailable() {
        let status = status_fixture();
        let mut value = serde_json::to_value(&status).unwrap();
        assert!(value["deployments"][0]["prefix_hits_total"].is_null());
        // Simulate a minor-0 worker: strip every optional routing signal.
        for field in [
            "kv_cache_usage_pct",
            "prefix_hits_total",
            "prefix_queries_total",
            "prefix_evictions_total",
            "kv_host_pages",
            "kv_demotions_total",
            "kv_promotions_total",
            "kv_promotion_latency_avg_seconds",
            "tokens_out_per_s_ema",
            "observation_cost_units",
        ] {
            value["deployments"][0]
                .as_object_mut()
                .unwrap()
                .remove(field);
        }
        value["schema_version"] = json!({ "major": 1, "minor": 0 });
        let decoded: WorkerStatus = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.schema_version, SchemaVersion::new(1, 0));
        let deployment = &decoded.deployments[0];
        assert!(deployment.kv_cache_usage_pct.is_none());
        assert!(deployment.prefix_hits_total.is_none());
        assert!(deployment.prefix_queries_total.is_none());
        assert!(deployment.prefix_evictions_total.is_none());
        assert!(deployment.kv_host_pages.is_none());
        assert!(deployment.kv_demotions_total.is_none());
        assert!(deployment.kv_promotions_total.is_none());
        assert!(deployment.kv_promotion_latency_avg_seconds.is_none());
        assert!(deployment.tokens_out_per_s_ema.is_none());
        assert!(deployment.observation_cost_units.is_none());
    }

    #[test]
    fn status_round_trips_present_routing_signals() {
        let mut status = status_fixture();
        let deployment = &mut status.deployments[0];
        deployment.kv_cache_usage_pct = Some(42.5);
        deployment.prefix_hits_total = Some(11);
        deployment.prefix_queries_total = Some(20);
        deployment.prefix_evictions_total = Some(2);
        deployment.kv_host_pages = Some(7);
        deployment.kv_demotions_total = Some(24);
        deployment.kv_promotions_total = Some(12);
        deployment.kv_promotion_latency_avg_seconds = Some(0.000_25);
        deployment.tokens_out_per_s_ema = Some(18.75);
        deployment.observation_cost_units = Some(1);
        let encoded = serde_json::to_vec(&status).unwrap();
        let decoded: WorkerStatus = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, status);
        assert_eq!(decoded.deployments[0].kv_cache_usage_pct, Some(42.5));
        assert_eq!(decoded.deployments[0].kv_host_pages, Some(7));
        assert_eq!(decoded.deployments[0].kv_promotions_total, Some(12));
        assert_eq!(decoded.deployments[0].tokens_out_per_s_ema, Some(18.75));
    }

    #[test]
    fn invocation_round_trips_and_validates() {
        let request = request_fixture();
        request.validate().unwrap();
        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: InvocationRequest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn validation_rejects_mismatched_task_and_unbounded_chat_fields() {
        let mut request = request_fixture();
        request.task = TaskKind::SpeechToText;
        assert!(matches!(
            request.validate(),
            Err(ContractValidationError::TaskMismatch { .. })
        ));

        let mut request = request_fixture();
        let InvocationInput::Chat { input, .. } = &mut request.input;
        input.messages[0].content = "x".repeat(MAX_CHAT_MESSAGE_BYTES + 1);
        assert!(matches!(
            request.validate(),
            Err(ContractValidationError::InvalidMessageLength { index: 0 })
        ));

        let mut request = request_fixture();
        request.remaining_time_ms = 0;
        assert!(matches!(
            request.validate(),
            Err(ContractValidationError::InvalidExecutionBudget)
        ));

        let mut request = request_fixture();
        request.remaining_time_ms = MAX_REMAINING_TIME_MS + 1;
        assert!(matches!(
            request.validate(),
            Err(ContractValidationError::InvalidExecutionBudget)
        ));
    }

    #[test]
    fn unknown_enum_variant_does_not_silently_downgrade() {
        assert!(serde_json::from_value::<BackendKind>(json!("rocm")).is_err());
        assert!(serde_json::from_value::<TaskKind>(json!("unknown_task")).is_err());
        assert!(serde_json::from_value::<RejectionCode>(json!("try_cpu_instead")).is_err());
    }

    #[test]
    fn rejection_constructor_is_unambiguously_not_accepted() {
        let rejection = InvocationRejection::new(
            id("req-1"),
            id("attempt-1"),
            RejectionCode::CapacityExhausted,
            "full",
        );
        assert!(!rejection.accepted);
        assert!(rejection.is_valid());
        assert_eq!(
            serde_json::to_value(rejection).unwrap()["code"],
            "capacity_exhausted"
        );
    }

    #[test]
    fn event_wire_shape_is_sequenced_and_terminal_is_explicit() {
        let event = InvocationEvent {
            schema_version: PROTOCOL_V1,
            request_id: id("req-1"),
            attempt_id: id("attempt-1"),
            sequence: 3,
            event: InvocationEventKind::Completed {
                finish_reason: FinishReason::Stop,
                usage: Some(Usage {
                    input_tokens: 2,
                    output_tokens: 4,
                    cached_tokens: None,
                }),
            },
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "completed");
        assert_eq!(value["sequence"], 3);
        assert!(event.is_terminal());
        assert_eq!(
            serde_json::from_value::<InvocationEvent>(value).unwrap(),
            event
        );
    }

    #[test]
    fn text_delta_logprobs_are_additive_and_absent_when_unmeasured() {
        let plain = InvocationEventKind::TextDelta {
            text: "hi".into(),
            logprobs: None,
        };
        let value = serde_json::to_value(&plain).unwrap();
        assert!(
            value.get("logprobs").is_none(),
            "absent logprobs must omit the field, got {value}"
        );

        let measured = InvocationEventKind::TextDelta {
            text: "hi".into(),
            logprobs: Some(vec![TokenLogprob {
                token: "hi".into(),
                logprob: -0.5,
                bytes: b"hi".to_vec(),
                top_logprobs: vec![TopTokenLogprob {
                    token: "hi".into(),
                    logprob: -0.5,
                    bytes: b"hi".to_vec(),
                }],
            }]),
        };
        let value = serde_json::to_value(&measured).unwrap();
        assert_eq!(value["logprobs"][0]["token"], "hi");
        assert_eq!(value["logprobs"][0]["bytes"], serde_json::json!([104, 105]));
        assert_eq!(value["logprobs"][0]["top_logprobs"][0]["logprob"], -0.5);
        assert_eq!(
            serde_json::from_value::<InvocationEventKind>(value).unwrap(),
            measured
        );

        // Minor-2 peers never send the field.
        let legacy = serde_json::json!({"type": "text_delta", "text": "hi"});
        assert_eq!(
            serde_json::from_value::<InvocationEventKind>(legacy).unwrap(),
            plain
        );
    }

    #[test]
    fn cached_tokens_is_additive_and_absent_when_unmeasured() {
        let without = Usage {
            input_tokens: 2,
            output_tokens: 4,
            cached_tokens: None,
        };
        let value = serde_json::to_value(&without).unwrap();
        assert!(
            value.get("cached_tokens").is_none(),
            "absent measurement must omit the field, got {value}"
        );

        let with = Usage {
            input_tokens: 2,
            output_tokens: 4,
            cached_tokens: Some(3),
        };
        let value = serde_json::to_value(&with).unwrap();
        assert_eq!(value["cached_tokens"], 3);
        assert_eq!(serde_json::from_value::<Usage>(value).unwrap(), with);

        // Minor-2 workers (and older gateways) never send the field.
        let legacy = serde_json::json!({"input_tokens": 2, "output_tokens": 4});
        assert_eq!(serde_json::from_value::<Usage>(legacy).unwrap(), without);
    }

    #[test]
    fn unknown_attempt_and_cancel_do_not_confirm_teardown() {
        assert!(!AttemptState::Unknown.proves_execution_stopped());
        assert!(!AttemptState::Expired.proves_execution_stopped());
        assert!(!AttemptState::ExecutionStopping.proves_execution_stopped());
        assert!(!CancelDisposition::Requested.confirms_execution_stopped());
        assert!(!CancelDisposition::Unknown.confirms_execution_stopped());
        assert!(CancelDisposition::Stopped.confirms_execution_stopped());
    }
}

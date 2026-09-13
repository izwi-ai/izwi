use crate::{ChildLaunchSpec, ReadinessPolicy, RestartPolicy, ShutdownPolicy, ValidatedNodeConfig};
use izwi_serving_client::{WorkerClient, WorkerClientConfig, WorkerClientError};
use izwi_serving_protocol::{
    ArtifactRevision, BackendKind, Capability, DeploymentId, DeviceAssignment, IncarnationId,
    ModelAlias, ModelGeneration, ModelReadiness, NodeId, ServiceCredentials, TaskKind,
    WorkerDescriptor, WorkerId, WorkerProcessState, WorkerStatus,
};
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, io, process::ExitStatus, time::Duration};
use tokio::{
    io::AsyncWriteExt,
    process::{Child, ChildStdin},
    time::Instant,
};

pub const MAX_READINESS_DIAGNOSTIC_BYTES: usize = 1024;
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Immutable identity that a freshly launched worker must prove before it is usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedWorkerIdentity {
    worker_id: WorkerId,
    node_id: NodeId,
    incarnation_id: IncarnationId,
    assignment: DeviceAssignment,
    deployment_id: DeploymentId,
    public_model: ModelAlias,
    artifact_revision: ArtifactRevision,
    model_generation: ModelGeneration,
    task: TaskKind,
    backend: BackendKind,
    precision: String,
    execution_representation: String,
    tokenizer_revision: Option<ArtifactRevision>,
    capability: Capability,
    max_active_invocations: u32,
    endpoint: String,
}

impl ExpectedWorkerIdentity {
    pub fn from_config(
        node: &ValidatedNodeConfig,
        worker_id: &WorkerId,
        incarnation_id: IncarnationId,
    ) -> Result<Self, LifecycleError> {
        let worker = node
            .worker(worker_id)
            .ok_or_else(|| LifecycleError::UnknownWorker(worker_id.clone()))?;
        Ok(Self {
            worker_id: worker.worker_id.clone(),
            node_id: node.config().node_id.clone(),
            incarnation_id,
            assignment: worker.assignment.clone(),
            deployment_id: worker.deployment.deployment_id.clone(),
            public_model: worker.deployment.public_model.clone(),
            artifact_revision: worker.deployment.artifact_revision.clone(),
            model_generation: worker.deployment.model_generation,
            task: worker.deployment.task,
            backend: worker.deployment.backend,
            precision: worker.deployment.precision.clone(),
            execution_representation: worker.deployment.execution_representation.clone(),
            tokenizer_revision: worker.deployment.tokenizer_revision.clone(),
            capability: worker.deployment.expected_capability(),
            max_active_invocations: worker.max_active_invocations,
            endpoint: format!("http://{}", worker.bind),
        })
    }

    pub fn worker_id(&self) -> &WorkerId {
        &self.worker_id
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn incarnation_id(&self) -> &IncarnationId {
        &self.incarnation_id
    }

    pub fn assignment(&self) -> &DeviceAssignment {
        &self.assignment
    }

    pub fn deployment_id(&self) -> &DeploymentId {
        &self.deployment_id
    }

    pub fn model_generation(&self) -> ModelGeneration {
        self.model_generation
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

/// Stateful verifier that rejects stale, replaced, or misassigned worker observations.
#[derive(Debug, Clone)]
pub struct ReadinessTracker {
    expected: ExpectedWorkerIdentity,
    descriptor_verified: bool,
    last_status_sequence: Option<u64>,
}

impl ReadinessTracker {
    pub fn new(expected: ExpectedWorkerIdentity) -> Self {
        Self {
            expected,
            descriptor_verified: false,
            last_status_sequence: None,
        }
    }

    pub fn expected(&self) -> &ExpectedWorkerIdentity {
        &self.expected
    }

    pub fn verify_descriptor(
        &mut self,
        descriptor: &WorkerDescriptor,
    ) -> Result<(), IdentityMismatch> {
        if descriptor.worker_id != self.expected.worker_id {
            return Err(IdentityMismatch::WorkerId);
        }
        if descriptor.node_id != self.expected.node_id {
            return Err(IdentityMismatch::NodeId);
        }
        if descriptor.incarnation_id != self.expected.incarnation_id {
            return Err(IdentityMismatch::Incarnation);
        }
        if descriptor.assignment != self.expected.assignment {
            return Err(IdentityMismatch::Assignment);
        }
        self.descriptor_verified = true;
        Ok(())
    }

    /// Returns true only after the exact assigned deployment reports ready.
    pub fn observe_status(&mut self, status: &WorkerStatus) -> Result<bool, IdentityMismatch> {
        if !self.descriptor_verified {
            return Err(IdentityMismatch::DescriptorNotVerified);
        }
        self.verify_status_identity(status)?;
        if status.status_sequence == 0
            || self
                .last_status_sequence
                .is_some_and(|last| status.status_sequence <= last)
        {
            return Err(IdentityMismatch::NonIncreasingStatusSequence);
        }
        self.last_status_sequence = Some(status.status_sequence);
        if status.capacity.max_active_invocations != self.expected.max_active_invocations
            || status.capacity.active_invocations > status.capacity.max_active_invocations
            || status.capacity.available_admission_credits > status.capacity.max_active_invocations
            || status
                .capacity
                .active_invocations
                .checked_add(status.capacity.available_admission_credits)
                != Some(status.capacity.max_active_invocations)
            || status.capacity.queued_invocations > status.capacity.max_queued_invocations
            || status.capacity.reserved_sessions > status.capacity.max_sessions
        {
            return Err(IdentityMismatch::Capacity);
        }

        let mut matching = status.deployments.iter().filter(|deployment| {
            deployment.deployment_id == self.expected.deployment_id
                && deployment.public_model == self.expected.public_model
                && deployment.artifact_revision == self.expected.artifact_revision
                && deployment.model_generation == self.expected.model_generation
                && deployment.backend == self.expected.backend
        });
        let Some(deployment) = matching.next() else {
            return Err(IdentityMismatch::Deployment);
        };
        if matching.next().is_some() {
            return Err(IdentityMismatch::DuplicateDeployment);
        }
        if deployment.task != self.expected.task {
            return Err(IdentityMismatch::DeploymentTask);
        }
        if deployment.precision != self.expected.precision
            || deployment.execution_representation != self.expected.execution_representation
            || deployment.tokenizer_revision != self.expected.tokenizer_revision
        {
            return Err(IdentityMismatch::ExecutionProfile);
        }
        if deployment.capability != self.expected.capability {
            return Err(IdentityMismatch::CapabilityProfile);
        }
        if deployment.readiness == ModelReadiness::Failed {
            return Err(IdentityMismatch::DeploymentFailed);
        }
        Ok(status.process_state == WorkerProcessState::Running
            && deployment.readiness == ModelReadiness::Ready)
    }

    fn verify_status_identity(&self, status: &WorkerStatus) -> Result<(), IdentityMismatch> {
        if status.worker_id != self.expected.worker_id {
            return Err(IdentityMismatch::WorkerId);
        }
        if status.node_id != self.expected.node_id {
            return Err(IdentityMismatch::NodeId);
        }
        if status.incarnation_id != self.expected.incarnation_id {
            return Err(IdentityMismatch::Incarnation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IdentityMismatch {
    #[error("worker id differs from the assigned worker")]
    WorkerId,
    #[error("node id differs from the assigned node")]
    NodeId,
    #[error("worker incarnation differs from the launched incarnation")]
    Incarnation,
    #[error("device assignment differs from the validated assignment")]
    Assignment,
    #[error("worker descriptor must be verified before accepting status")]
    DescriptorNotVerified,
    #[error("worker status sequence did not increase within the incarnation")]
    NonIncreasingStatusSequence,
    #[error("worker capacity differs from its configured capacity or is inconsistent")]
    Capacity,
    #[error("worker did not report the exact assigned deployment generation")]
    Deployment,
    #[error("worker reported the assigned deployment more than once")]
    DuplicateDeployment,
    #[error("the assigned deployment task differs from the configured task")]
    DeploymentTask,
    #[error("the assigned deployment execution profile differs from its configured profile")]
    ExecutionProfile,
    #[error("the assigned deployment capability differs from its configured capability profile")]
    CapabilityProfile,
    #[error("the assigned deployment reported a failed readiness state")]
    DeploymentFailed,
}

/// A child process plus the only parent-side writer for its ownership pipe.
///
/// Dropping this value closes stdin and asks a managed worker to fail closed. Tokio's
/// kill-on-drop is also enabled as a final local-process ownership fence.
pub struct SupervisedWorker {
    child: Child,
    ownership_stdin: Option<ChildStdin>,
    client: WorkerClient,
    readiness: ReadinessTracker,
}

impl std::fmt::Debug for SupervisedWorker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SupervisedWorker")
            .field("process_id", &self.child.id())
            .field("expected", self.readiness.expected())
            .field("owns_control_pipe", &self.ownership_stdin.is_some())
            .finish_non_exhaustive()
    }
}

impl SupervisedWorker {
    pub fn spawn(
        spec: &ChildLaunchSpec,
        expected: ExpectedWorkerIdentity,
        credentials: ServiceCredentials,
        client_config: WorkerClientConfig,
    ) -> Result<Self, LifecycleError> {
        if !spec.pipes_stdin_for_ownership() {
            return Err(LifecycleError::OwnershipPipeNotConfigured);
        }
        let client = WorkerClient::new(&expected.endpoint, credentials, client_config)
            .map_err(LifecycleError::Client)?;
        let mut command = tokio::process::Command::from(spec.command());
        command.kill_on_drop(true);
        let mut child = command.spawn().map_err(LifecycleError::Spawn)?;
        let ownership_stdin = child
            .stdin
            .take()
            .ok_or(LifecycleError::OwnershipPipeUnavailable)?;
        Ok(Self {
            child,
            ownership_stdin: Some(ownership_stdin),
            client,
            readiness: ReadinessTracker::new(expected),
        })
    }

    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    pub fn client(&self) -> &WorkerClient {
        &self.client
    }

    pub fn readiness(&self) -> &ReadinessTracker {
        &self.readiness
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>, LifecycleError> {
        self.child.try_wait().map_err(LifecycleError::ProcessIo)
    }

    pub async fn wait_for_exit(&mut self) -> Result<ExitStatus, LifecycleError> {
        self.child.wait().await.map_err(LifecycleError::ProcessIo)
    }

    /// Idempotently closes the ownership pipe. Managed workers interpret EOF as
    /// an instruction to stop admission and enter their bounded shutdown policy.
    pub async fn begin_drain(&mut self) -> Result<(), LifecycleError> {
        if let Some(mut ownership_stdin) = self.ownership_stdin.take() {
            ownership_stdin
                .shutdown()
                .await
                .map_err(LifecycleError::ProcessIo)?;
            drop(ownership_stdin);
        }
        Ok(())
    }

    pub async fn wait_until_ready(
        &mut self,
        policy: &ReadinessPolicy,
    ) -> Result<WorkerStatus, LifecycleError> {
        let deadline = Instant::now() + Duration::from_millis(policy.startup_timeout_ms);
        let poll_interval = Duration::from_millis(policy.poll_interval_ms);
        let mut last_diagnostic = None;

        loop {
            if let Some(status) = self.child.try_wait().map_err(LifecycleError::ProcessIo)? {
                return Err(LifecycleError::ExitedBeforeReady(status));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(LifecycleError::ReadinessTimeout { last_diagnostic });
            }

            if !self.readiness.descriptor_verified {
                match tokio::time::timeout(remaining, self.client.descriptor()).await {
                    Ok(Ok(descriptor)) => self
                        .readiness
                        .verify_descriptor(&descriptor)
                        .map_err(LifecycleError::Identity)?,
                    Ok(Err(error)) if readiness_error_is_retryable(&error) => {
                        last_diagnostic = Some(bounded_diagnostic(&error));
                    }
                    Ok(Err(error)) => return Err(LifecycleError::Client(error)),
                    Err(_) => return Err(LifecycleError::ReadinessTimeout { last_diagnostic }),
                }
            }

            if self.readiness.descriptor_verified {
                let remaining = deadline.saturating_duration_since(Instant::now());
                match tokio::time::timeout(remaining, self.client.status()).await {
                    Ok(Ok(status)) => {
                        if self
                            .readiness
                            .observe_status(&status)
                            .map_err(LifecycleError::Identity)?
                        {
                            return Ok(status);
                        }
                    }
                    Ok(Err(error)) if readiness_error_is_retryable(&error) => {
                        last_diagnostic = Some(bounded_diagnostic(&error));
                    }
                    Ok(Err(error)) => return Err(LifecycleError::Client(error)),
                    Err(_) => return Err(LifecycleError::ReadinessTimeout { last_diagnostic }),
                }
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(LifecycleError::ReadinessTimeout { last_diagnostic });
            }
            tokio::time::sleep(poll_interval.min(remaining)).await;
        }
    }

    /// Closes the parent-owned pipe, waits for cooperative drain/cancellation,
    /// then escalates to TERM and finally KILL according to the validated policy.
    pub async fn drain_and_stop(
        mut self,
        policy: &ShutdownPolicy,
    ) -> Result<StopReport, LifecycleError> {
        self.begin_drain().await?;

        let cooperative = Duration::from_millis(policy.drain_grace_ms)
            .saturating_add(Duration::from_millis(policy.cancellation_grace_ms));
        let cooperative_deadline = Instant::now() + cooperative;
        let mut last_process_state = None;
        let mut identity_violation = None;
        while Instant::now() < cooperative_deadline {
            if let Some(exit_status) = self.child.try_wait().map_err(LifecycleError::ProcessIo)? {
                return Ok(StopReport {
                    outcome: StopOutcome::Cooperative,
                    exit_status,
                    last_process_state,
                    identity_violation,
                });
            }
            let remaining = cooperative_deadline.saturating_duration_since(Instant::now());
            let observation_budget = STOP_POLL_INTERVAL.min(remaining);
            if !observation_budget.is_zero() {
                if let Ok(Ok(status)) =
                    tokio::time::timeout(observation_budget, self.client.status()).await
                {
                    if let Err(error) = self.readiness.verify_status_identity(&status) {
                        identity_violation = Some(error);
                    } else {
                        last_process_state = Some(status.process_state);
                    }
                }
            }
            let remaining = cooperative_deadline.saturating_duration_since(Instant::now());
            if !remaining.is_zero() {
                tokio::time::sleep(STOP_POLL_INTERVAL.min(remaining)).await;
            }
        }

        request_termination(&mut self.child)?;
        let termination_deadline =
            Instant::now() + Duration::from_millis(policy.termination_grace_ms);
        while Instant::now() < termination_deadline {
            if let Some(exit_status) = self.child.try_wait().map_err(LifecycleError::ProcessIo)? {
                return Ok(StopReport {
                    outcome: StopOutcome::Terminated,
                    exit_status,
                    last_process_state,
                    identity_violation,
                });
            }
            let remaining = termination_deadline.saturating_duration_since(Instant::now());
            tokio::time::sleep(STOP_POLL_INTERVAL.min(remaining)).await;
        }

        self.child.kill().await.map_err(LifecycleError::ProcessIo)?;
        let exit_status = self.child.wait().await.map_err(LifecycleError::ProcessIo)?;
        Ok(StopReport {
            outcome: StopOutcome::Killed,
            exit_status,
            last_process_state,
            identity_violation,
        })
    }
}

#[derive(Debug)]
pub struct StopReport {
    pub outcome: StopOutcome,
    pub exit_status: ExitStatus,
    pub last_process_state: Option<WorkerProcessState>,
    pub identity_violation: Option<IdentityMismatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOutcome {
    Cooperative,
    Terminated,
    Killed,
}

#[derive(Debug, thiserror::Error)]
pub enum LifecycleError {
    #[error("unknown worker {0}")]
    UnknownWorker(WorkerId),
    #[error("worker launch specification does not configure a parent-owned stdin pipe")]
    OwnershipPipeNotConfigured,
    #[error("spawned worker did not expose its configured stdin ownership pipe")]
    OwnershipPipeUnavailable,
    #[error("failed to construct worker client: {0}")]
    Client(#[source] WorkerClientError),
    #[error("failed to spawn worker process: {0}")]
    Spawn(#[source] io::Error),
    #[error("worker process operation failed: {0}")]
    ProcessIo(#[source] io::Error),
    #[error("worker exited before readiness: {0}")]
    ExitedBeforeReady(ExitStatus),
    #[error("worker did not become ready before its startup deadline; last observation: {last_diagnostic:?}")]
    ReadinessTimeout { last_diagnostic: Option<String> },
    #[error("worker identity verification failed: {0}")]
    Identity(#[source] IdentityMismatch),
}

fn readiness_error_is_retryable(error: &WorkerClientError) -> bool {
    match error {
        WorkerClientError::Transport(_) | WorkerClientError::Deadline(_) => true,
        WorkerClientError::HttpStatus { status, .. } => status.is_server_error(),
        _ => false,
    }
}

fn bounded_diagnostic(error: &WorkerClientError) -> String {
    let mut message = error.to_string();
    if message.len() > MAX_READINESS_DIAGNOSTIC_BYTES {
        let mut boundary = MAX_READINESS_DIAGNOSTIC_BYTES;
        while !message.is_char_boundary(boundary) {
            boundary -= 1;
        }
        message.truncate(boundary);
    }
    message
}

#[cfg(unix)]
fn request_termination(child: &mut Child) -> Result<(), LifecycleError> {
    let Some(process_id) = child.id() else {
        return Ok(());
    };
    let process_id = i32::try_from(process_id).map_err(|_| {
        LifecycleError::ProcessIo(io::Error::new(
            io::ErrorKind::InvalidData,
            "child process id exceeds the platform range",
        ))
    })?;
    // SAFETY: `process_id` was returned for this live child and SIGTERM does not
    // dereference memory. A concurrent exit is treated as a successful stop.
    if unsafe { libc::kill(process_id, libc::SIGTERM) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(LifecycleError::ProcessIo(error))
    }
}

#[cfg(not(unix))]
fn request_termination(child: &mut Child) -> Result<(), LifecycleError> {
    child.start_kill().map_err(LifecycleError::ProcessIo)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartDecision {
    RestartAfter(Duration),
    Quarantine,
}

/// Bounded crash history and deterministic per-worker jitter.
#[derive(Debug, Clone)]
pub struct RestartController {
    policy: RestartPolicy,
    failures: VecDeque<Duration>,
    failure_streak: u32,
    jitter_seed: u64,
    jitter_sequence: u64,
    quarantined: bool,
}

impl RestartController {
    pub fn for_worker(
        node: &ValidatedNodeConfig,
        worker_id: &WorkerId,
    ) -> Result<Self, LifecycleError> {
        if node.worker(worker_id).is_none() {
            return Err(LifecycleError::UnknownWorker(worker_id.clone()));
        }
        let digest = Sha256::digest(worker_id.as_str().as_bytes());
        let jitter_seed = u64::from_le_bytes(digest[..8].try_into().expect("fixed digest width"));
        Ok(Self {
            policy: node.config().restart.clone(),
            failures: VecDeque::with_capacity(
                usize::from(node.config().restart.max_restarts_per_window).min(100),
            ),
            failure_streak: 0,
            jitter_seed,
            jitter_sequence: 0,
            quarantined: false,
        })
    }

    #[cfg(test)]
    fn with_seed(policy: RestartPolicy, jitter_seed: u64) -> Self {
        Self {
            failures: VecDeque::with_capacity(usize::from(policy.max_restarts_per_window)),
            policy,
            failure_streak: 0,
            jitter_seed,
            jitter_sequence: 0,
            quarantined: false,
        }
    }

    pub fn record_failure(&mut self, now: Duration, worker_uptime: Duration) -> RestartDecision {
        if self.quarantined {
            return RestartDecision::Quarantine;
        }
        if worker_uptime >= Duration::from_millis(self.policy.stable_reset_ms) {
            self.failures.clear();
            self.failure_streak = 0;
        }
        let window = Duration::from_millis(self.policy.restart_window_ms);
        while self
            .failures
            .front()
            .is_some_and(|recorded| now.saturating_sub(*recorded) >= window)
        {
            self.failures.pop_front();
        }
        if self.failures.len() >= usize::from(self.policy.max_restarts_per_window) {
            self.quarantined = true;
            return RestartDecision::Quarantine;
        }
        self.failures.push_back(now);

        let exponent = self.failure_streak.min(63);
        let multiplier = 1_u64.checked_shl(exponent).unwrap_or(u64::MAX);
        let base = self
            .policy
            .initial_backoff_ms
            .saturating_mul(multiplier)
            .min(self.policy.maximum_backoff_ms);
        self.failure_streak = self.failure_streak.saturating_add(1);
        let delay = jittered_delay_ms(
            base,
            self.policy.maximum_backoff_ms,
            self.policy.jitter_percent,
            splitmix64(self.jitter_seed ^ self.jitter_sequence),
        );
        self.jitter_sequence = self.jitter_sequence.wrapping_add(1);
        RestartDecision::RestartAfter(Duration::from_millis(delay))
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn recent_failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Quarantine is fail-closed and can only be cleared by an explicit operator action.
    pub fn clear_quarantine(&mut self) {
        self.failures.clear();
        self.failure_streak = 0;
        self.quarantined = false;
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn jittered_delay_ms(base: u64, maximum: u64, percent: u8, sample: u64) -> u64 {
    let span = u64::try_from((u128::from(base) * u128::from(percent)) / 100).unwrap_or(u64::MAX);
    if span == 0 {
        return base.min(maximum);
    }
    let width = span.saturating_mul(2).saturating_add(1);
    let offset = i128::from(sample % width) - i128::from(span);
    let adjusted = (i128::from(base) + offset).max(1);
    u64::try_from(adjusted).unwrap_or(maximum).min(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_protocol::{
        CancellationBehavior, Capability, CapacitySnapshot, InputFormat, LoadedDeployment,
        OutputFormat, SchemaVersion, TaskKind, WorkerFeature, PROTOCOL_V1,
    };
    use std::collections::BTreeSet;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn expected() -> ExpectedWorkerIdentity {
        ExpectedWorkerIdentity {
            worker_id: id("worker-a"),
            node_id: id("node-a"),
            incarnation_id: id("incarnation-a"),
            assignment: DeviceAssignment::Cpu {
                thread_budget: 2,
                affinity: vec![0, 1],
                host_memory_limit_bytes: 4096,
            },
            deployment_id: id("deployment-a"),
            public_model: id("model-a"),
            artifact_revision: id("revision-a"),
            model_generation: ModelGeneration::new(7).unwrap(),
            task: TaskKind::Chat,
            backend: BackendKind::Cpu,
            precision: "f32".into(),
            execution_representation: "dense".into(),
            tokenizer_revision: None,
            capability: Capability {
                task: TaskKind::Chat,
                streaming: true,
                realtime: false,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                output_formats: BTreeSet::from([OutputFormat::Text]),
                max_input_bytes: 4096,
                max_context_tokens: Some(512),
                max_output_tokens: Some(128),
            },
            max_active_invocations: 2,
            endpoint: "http://127.0.0.1:9470".into(),
        }
    }

    fn descriptor(expected: &ExpectedWorkerIdentity) -> WorkerDescriptor {
        WorkerDescriptor {
            schema_version: PROTOCOL_V1,
            supported_protocol_versions: vec![SchemaVersion::new(1, 0)],
            worker_id: expected.worker_id.clone(),
            node_id: expected.node_id.clone(),
            incarnation_id: expected.incarnation_id.clone(),
            build_version: "test".into(),
            assignment: expected.assignment.clone(),
            features: BTreeSet::from([WorkerFeature::Streaming]),
        }
    }

    fn status(expected: &ExpectedWorkerIdentity, sequence: u64) -> WorkerStatus {
        WorkerStatus {
            schema_version: PROTOCOL_V1,
            worker_id: expected.worker_id.clone(),
            node_id: expected.node_id.clone(),
            incarnation_id: expected.incarnation_id.clone(),
            status_sequence: sequence,
            process_state: WorkerProcessState::Running,
            deployments: vec![LoadedDeployment {
                deployment_id: expected.deployment_id.clone(),
                public_model: expected.public_model.clone(),
                artifact_revision: expected.artifact_revision.clone(),
                model_generation: expected.model_generation,
                task: expected.task,
                backend: expected.backend,
                precision: expected.precision.clone(),
                execution_representation: expected.execution_representation.clone(),
                tokenizer_revision: expected.tokenizer_revision.clone(),
                readiness: ModelReadiness::Ready,
                capability: expected.capability.clone(),
            }],
            capacity: CapacitySnapshot {
                max_active_invocations: expected.max_active_invocations,
                active_invocations: 0,
                max_queued_invocations: 0,
                queued_invocations: 0,
                max_sessions: 0,
                reserved_sessions: 0,
                available_admission_credits: expected.max_active_invocations,
                outstanding_cost_units: 0,
            },
        }
    }

    #[test]
    fn readiness_requires_exact_incarnation_assignment_and_deployment() {
        let expected = expected();
        let mut tracker = ReadinessTracker::new(expected.clone());
        tracker.verify_descriptor(&descriptor(&expected)).unwrap();
        assert!(tracker.observe_status(&status(&expected, 1)).unwrap());

        let mut stale = status(&expected, 2);
        stale.incarnation_id = id("old-incarnation");
        assert_eq!(
            tracker.observe_status(&stale).unwrap_err(),
            IdentityMismatch::Incarnation
        );
    }

    #[test]
    fn readiness_rejects_assignment_generation_and_capacity_changes() {
        let expected = expected();
        let mut wrong_descriptor = descriptor(&expected);
        wrong_descriptor.assignment = DeviceAssignment::Cpu {
            thread_budget: 1,
            affinity: vec![0],
            host_memory_limit_bytes: 4096,
        };
        assert_eq!(
            ReadinessTracker::new(expected.clone())
                .verify_descriptor(&wrong_descriptor)
                .unwrap_err(),
            IdentityMismatch::Assignment
        );

        let mut tracker = ReadinessTracker::new(expected.clone());
        tracker.verify_descriptor(&descriptor(&expected)).unwrap();
        let mut wrong_generation = status(&expected, 1);
        wrong_generation.deployments[0].model_generation = ModelGeneration::new(8).unwrap();
        assert_eq!(
            tracker.observe_status(&wrong_generation).unwrap_err(),
            IdentityMismatch::Deployment
        );

        let mut wrong_capacity = status(&expected, 2);
        wrong_capacity.capacity.available_admission_credits = 1;
        assert_eq!(
            tracker.observe_status(&wrong_capacity).unwrap_err(),
            IdentityMismatch::Capacity
        );
    }

    #[test]
    fn readiness_rejects_non_increasing_status_sequence() {
        let expected = expected();
        let mut tracker = ReadinessTracker::new(expected.clone());
        tracker.verify_descriptor(&descriptor(&expected)).unwrap();
        tracker.observe_status(&status(&expected, 5)).unwrap();
        assert_eq!(
            tracker.observe_status(&status(&expected, 5)).unwrap_err(),
            IdentityMismatch::NonIncreasingStatusSequence
        );
    }

    #[test]
    fn readiness_rejects_task_and_execution_profile_mismatches() {
        let expected = expected();
        let mut tracker = ReadinessTracker::new(expected.clone());
        tracker.verify_descriptor(&descriptor(&expected)).unwrap();

        let mut wrong_task = status(&expected, 1);
        wrong_task.deployments[0].task = TaskKind::SpeechToText;
        assert_eq!(
            tracker.observe_status(&wrong_task).unwrap_err(),
            IdentityMismatch::DeploymentTask
        );

        let mut wrong_precision = status(&expected, 2);
        wrong_precision.deployments[0].precision = "f16".into();
        assert_eq!(
            tracker.observe_status(&wrong_precision).unwrap_err(),
            IdentityMismatch::ExecutionProfile
        );
    }

    #[test]
    fn readiness_rejects_exact_capability_profile_mismatches() {
        let expected = expected();
        let mut tracker = ReadinessTracker::new(expected.clone());
        tracker.verify_descriptor(&descriptor(&expected)).unwrap();

        let mut wrong_capability_task = status(&expected, 1);
        wrong_capability_task.deployments[0].capability.task = TaskKind::TextToSpeech;
        assert_eq!(
            tracker.observe_status(&wrong_capability_task).unwrap_err(),
            IdentityMismatch::CapabilityProfile
        );

        let mut wrong_formats = status(&expected, 2);
        wrong_formats.deployments[0]
            .capability
            .output_formats
            .insert(OutputFormat::Json);
        assert_eq!(
            tracker.observe_status(&wrong_formats).unwrap_err(),
            IdentityMismatch::CapabilityProfile
        );
    }

    fn restart_policy() -> RestartPolicy {
        RestartPolicy {
            initial_backoff_ms: 100,
            maximum_backoff_ms: 800,
            restart_window_ms: 1_000,
            stable_reset_ms: 5_000,
            max_restarts_per_window: 3,
            jitter_percent: 0,
        }
    }

    #[test]
    fn restart_backoff_is_bounded_and_quarantines_at_budget() {
        let mut controller = RestartController::with_seed(restart_policy(), 7);
        assert_eq!(
            controller.record_failure(Duration::ZERO, Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(100))
        );
        assert_eq!(
            controller.record_failure(Duration::from_millis(10), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(200))
        );
        assert_eq!(
            controller.record_failure(Duration::from_millis(20), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(400))
        );
        assert_eq!(
            controller.record_failure(Duration::from_millis(30), Duration::ZERO),
            RestartDecision::Quarantine
        );
        assert!(controller.is_quarantined());
        assert_eq!(controller.recent_failure_count(), 3);
    }

    #[test]
    fn stable_uptime_resets_backoff_and_expired_failures_leave_window() {
        let mut controller = RestartController::with_seed(restart_policy(), 7);
        let _ = controller.record_failure(Duration::ZERO, Duration::ZERO);
        let _ = controller.record_failure(Duration::from_millis(10), Duration::ZERO);
        assert_eq!(
            controller.record_failure(Duration::from_millis(2_000), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(400))
        );
        assert_eq!(controller.recent_failure_count(), 1);
        assert_eq!(
            controller.record_failure(Duration::from_millis(2_100), Duration::from_secs(5)),
            RestartDecision::RestartAfter(Duration::from_millis(100))
        );
    }

    #[test]
    fn deterministic_jitter_never_exceeds_maximum() {
        let mut policy = restart_policy();
        policy.jitter_percent = 25;
        let mut first = RestartController::with_seed(policy.clone(), 42);
        let mut second = RestartController::with_seed(policy, 42);
        for index in 0..3 {
            let now = Duration::from_millis(index * 10);
            let left = first.record_failure(now, Duration::ZERO);
            let right = second.record_failure(now, Duration::ZERO);
            assert_eq!(left, right);
            let RestartDecision::RestartAfter(delay) = left else {
                panic!("restart budget unexpectedly exhausted")
            };
            assert!(delay <= Duration::from_millis(800));
            assert!(!delay.is_zero());
        }
    }
}

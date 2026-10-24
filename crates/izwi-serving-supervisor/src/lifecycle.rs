use crate::{ChildLaunchSpec, ReadinessPolicy, ShutdownPolicy, ValidatedNodeConfig};
use izwi_serving_client::{WorkerClient, WorkerClientConfig, WorkerClientError};
use izwi_serving_protocol::{
    ArtifactRevision, BackendKind, Capability, DeploymentId, DeviceAssignment, IncarnationId,
    LoadedDeployment, ModelAlias, ModelGeneration, ModelReadiness, NodeId, ServiceCredentials,
    TaskKind, WorkerDescriptor, WorkerId, WorkerProcessState, WorkerStatus,
};
use std::{io, process::ExitStatus, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin},
    time::Instant,
};

pub const MAX_READINESS_DIAGNOSTIC_BYTES: usize = 1024;
/// A worker output line longer than this is relayed in pieces.
const MAX_RELAYED_LINE_BYTES: usize = 64 * 1024;
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
    /// `None` accepts the capacity the worker derived from its model.
    max_active_invocations: Option<u32>,
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

    /// Checks the task, execution profile, and capability a worker reports for
    /// the assigned deployment. Readiness is observed state and not compared.
    pub fn verify_deployment(&self, deployment: &LoadedDeployment) -> Result<(), IdentityMismatch> {
        if deployment.task != self.task {
            return Err(IdentityMismatch::DeploymentTask);
        }
        if deployment.precision != self.precision
            || deployment.execution_representation != self.execution_representation
            || deployment.tokenizer_revision != self.tokenizer_revision
        {
            return Err(IdentityMismatch::ExecutionProfile);
        }
        if deployment.capability != self.capability {
            return Err(IdentityMismatch::CapabilityProfile);
        }
        Ok(())
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
        if status.capacity.max_active_invocations == 0
            || self
                .expected
                .max_active_invocations
                .is_some_and(|expected| status.capacity.max_active_invocations != expected)
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
        self.expected.verify_deployment(deployment)?;
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

/// Annotate one worker output line with its worker id: a JSON object line
/// gains a leading `"worker_id"` field and stays JSON; any other line gets a
/// `worker_id | ` prefix.
pub fn annotate_worker_line(worker_id: &str, line: &[u8]) -> Vec<u8> {
    let id = serde_json::to_string(worker_id).expect("a string serializes");
    let trimmed = line.trim_ascii_start();
    let mut annotated = Vec::with_capacity(line.len() + id.len() + 16);
    if let Some(rest) = trimmed.strip_prefix(b"{") {
        annotated.extend_from_slice(b"{\"worker_id\":");
        annotated.extend_from_slice(id.as_bytes());
        if !rest.trim_ascii_start().starts_with(b"}") {
            annotated.push(b',');
        }
        annotated.extend_from_slice(rest);
    } else {
        annotated.extend_from_slice(worker_id.as_bytes());
        annotated.extend_from_slice(b" | ");
        annotated.extend_from_slice(line);
    }
    if !annotated.ends_with(b"\n") {
        annotated.push(b'\n');
    }
    annotated
}

/// Copy a worker's stdout or stderr to the supervisor's own, one annotated
/// line at a time, until the worker closes it.
async fn relay_worker_output<R>(output: R, worker_id: String, to_stderr: bool)
where
    R: tokio::io::AsyncRead + Unpin,
{
    use std::io::Write as _;
    use tokio::io::AsyncBufReadExt;
    let mut reader = tokio::io::BufReader::new(output);
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_RELAYED_LINE_BYTES as u64)
            .read_until(b'\n', &mut line)
            .await;
        match read {
            Ok(0) | Err(_) => return,
            Ok(_) => {
                let annotated = annotate_worker_line(&worker_id, &line);
                let _ = if to_stderr {
                    std::io::stderr().lock().write_all(&annotated)
                } else {
                    std::io::stdout().lock().write_all(&annotated)
                };
            }
        }
    }
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
        command
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = command.spawn().map_err(LifecycleError::Spawn)?;
        let ownership_stdin = child
            .stdin
            .take()
            .ok_or(LifecycleError::OwnershipPipeUnavailable)?;
        let worker_id = expected.worker_id().as_str();
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(relay_worker_output(stdout, worker_id.to_string(), false));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(relay_worker_output(stderr, worker_id.to_string(), true));
        }
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

    /// Stops a worker that no longer answers: closes the pipe, sends TERM at
    /// once, waits the termination grace, then kills it. There is no
    /// cooperative drain, since the worker cannot be asked anything.
    pub async fn terminate(
        mut self,
        policy: &ShutdownPolicy,
    ) -> Result<StopReport, LifecycleError> {
        drop(self.ownership_stdin.take());
        request_termination(&mut self.child)?;
        let deadline = Instant::now() + Duration::from_millis(policy.termination_grace_ms);
        while Instant::now() < deadline {
            if let Some(exit_status) = self.child.try_wait().map_err(LifecycleError::ProcessIo)? {
                return Ok(StopReport {
                    outcome: StopOutcome::Terminated,
                    exit_status,
                    last_process_state: None,
                    identity_violation: None,
                });
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            tokio::time::sleep(STOP_POLL_INTERVAL.min(remaining)).await;
        }
        self.child.kill().await.map_err(LifecycleError::ProcessIo)?;
        let exit_status = self.child.wait().await.map_err(LifecycleError::ProcessIo)?;
        Ok(StopReport {
            outcome: StopOutcome::Killed,
            exit_status,
            last_process_state: None,
            identity_violation: None,
        })
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
        WorkerClientError::ConnectionNotEstablished(_)
        | WorkerClientError::Transport(_)
        | WorkerClientError::Deadline(_) => true,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_output_lines_carry_their_worker_id() {
        assert_eq!(
            annotate_worker_line("worker-a", b"plain text line\n"),
            b"worker-a | plain text line\n"
        );
        // A JSON line stays one JSON object with the id as a field.
        let json = annotate_worker_line("worker-a", br#"{"level":"INFO","message":"ready"}"#);
        let value: serde_json::Value = serde_json::from_slice(&json).expect("still JSON");
        assert_eq!(value["worker_id"], "worker-a");
        assert_eq!(value["message"], "ready");
        assert!(json.ends_with(b"\n"));
        let empty = annotate_worker_line("w\"q", b"{}\n");
        let value: serde_json::Value = serde_json::from_slice(&empty).expect("JSON");
        assert_eq!(value["worker_id"], "w\"q");
    }
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
            max_active_invocations: Some(2),
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
                kv_cache_usage_pct: None,
                prefix_hits_total: None,
                prefix_queries_total: None,
                prefix_evictions_total: None,
                tokens_out_per_s_ema: None,
                observation_cost_units: None,
                kv_host_pages: None,
                kv_demotions_total: None,
                kv_promotions_total: None,
                kv_promotion_latency_avg_seconds: None,
            }],
            capacity: CapacitySnapshot {
                max_active_invocations: expected.max_active_invocations.unwrap_or(1),
                active_invocations: 0,
                max_queued_invocations: 0,
                queued_invocations: 0,
                max_sessions: 0,
                reserved_sessions: 0,
                available_admission_credits: expected.max_active_invocations.unwrap_or(1),
                outstanding_cost_units: 0,
            },
        }
    }

    #[test]
    fn liveness_judges_identity_timeouts_and_failed_runtimes() {
        use crate::probe::{judge, ProbeVerdict};
        let expected = expected();
        assert_eq!(
            judge(&expected, Some(Ok(status(&expected, 9)))),
            ProbeVerdict::Alive
        );
        assert!(matches!(judge(&expected, None), ProbeVerdict::Failed(_)));
        let mut replaced = status(&expected, 9);
        replaced.incarnation_id = id("incarnation-b");
        assert!(matches!(
            judge(&expected, Some(Ok(replaced))),
            ProbeVerdict::Failed(_)
        ));
        let mut failed = status(&expected, 9);
        failed.process_state = WorkerProcessState::Failed;
        assert!(matches!(
            judge(&expected, Some(Ok(failed))),
            ProbeVerdict::Failed(_)
        ));
        // A deployment that is merely not ready is a readiness matter.
        let mut degraded = status(&expected, 9);
        degraded.deployments[0].readiness = ModelReadiness::Failed;
        assert_eq!(judge(&expected, Some(Ok(degraded))), ProbeVerdict::Alive);
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
    fn readiness_accepts_a_derived_capacity_only_when_none_is_declared() {
        // Declared: the worker must advertise exactly that capacity.
        let declared = expected();
        let mut tracker = ReadinessTracker::new(declared.clone());
        tracker.verify_descriptor(&descriptor(&declared)).unwrap();
        let mut derived = status(&declared, 1);
        derived.capacity.max_active_invocations = 8;
        derived.capacity.available_admission_credits = 8;
        assert_eq!(
            tracker.observe_status(&derived).unwrap_err(),
            IdentityMismatch::Capacity
        );

        // Undeclared: the worker sized itself from its model; any positive,
        // self-consistent capacity is accepted, and zero never is.
        let mut automatic = expected();
        automatic.max_active_invocations = None;
        let mut tracker = ReadinessTracker::new(automatic.clone());
        tracker.verify_descriptor(&descriptor(&automatic)).unwrap();
        let mut derived = status(&automatic, 1);
        derived.capacity.max_active_invocations = 8;
        derived.capacity.available_admission_credits = 8;
        assert!(tracker.observe_status(&derived).unwrap());
        let mut zero = status(&automatic, 2);
        zero.capacity.max_active_invocations = 0;
        zero.capacity.available_admission_credits = 0;
        assert_eq!(
            tracker.observe_status(&zero).unwrap_err(),
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
}

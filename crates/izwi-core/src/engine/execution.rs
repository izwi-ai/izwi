//! Authoritative execution plans, reports, capabilities, and lifecycle states.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::resources::{ResourceEstimate, ResourceReservation, ResourceVector};
use super::{RequestId, SequenceId, TaskType};

pub type PlanId = u64;
pub type SessionEpoch = SequenceId;

/// Identity of one request incarnation. Public request IDs may be reused after
/// completion, so executor transactions must also carry the scheduler epoch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    pub request_id: RequestId,
    pub epoch: SessionEpoch,
}

impl SessionKey {
    pub fn new(request_id: RequestId, epoch: SessionEpoch) -> Self {
        Self { request_id, epoch }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InputRange {
    pub start: usize,
    pub end: usize,
}

impl InputRange {
    pub fn new(start: usize, end: usize) -> Result<Self> {
        if end < start {
            return Err(Error::InvalidInput(
                "execution input range is reversed".to_string(),
            ));
        }
        Ok(Self { start, end })
    }

    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequencePhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkUnit {
    SequenceStep {
        phase: SequencePhase,
        input: InputRange,
        max_output_steps: usize,
    },
    AtomicJob {
        kind: String,
    },
    PipelineStage {
        name: String,
        ordinal: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionCapabilities {
    pub incremental_prefill: bool,
    pub incremental_decode: bool,
    pub native_batch: bool,
    pub mixed_phase_batch: bool,
    pub cancellable_between_steps: bool,
    pub recompute_safe: bool,
    pub cache_release_safe: bool,
    pub physical_cache: bool,
    pub max_batch_size: usize,
}

/// The unit of work an executor actually exposes for this request.
///
/// This is intentionally separate from the public capability (chat, ASR,
/// TTS, ...): two models serving the same capability can have very different
/// execution and cancellation semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Sequence,
    Atomic,
    Realtime,
    Pipeline,
    Artifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillMode {
    None,
    Full,
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeBatchMode {
    None,
    Static,
    Continuous,
}

/// Observed dispatch mechanism for one executor report. Request-parallel work
/// is intentionally distinct from a model tensor batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchDispatchKind {
    #[default]
    Serial,
    RequestParallel,
    TensorStatic,
    TensorContinuous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchDispatch {
    pub kind: BatchDispatchKind,
    pub width: usize,
}

impl BatchDispatch {
    pub const fn serial() -> Self {
        Self {
            kind: BatchDispatchKind::Serial,
            width: 1,
        }
    }

    pub const fn new(kind: BatchDispatchKind, width: usize) -> Self {
        Self { kind, width }
    }
}

impl Default for BatchDispatch {
    fn default() -> Self {
        Self::serial()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    None,
    OpaqueModelOwned,
    ExternalPaged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationGranularity {
    OperationBoundary,
    SequenceStep,
    RealtimeChunk,
    PipelineStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyClass {
    Exclusive,
    Batchable,
}

/// Effective execution behavior for one model/request/backend combination.
///
/// Profiles fail closed: advanced scheduling features stay disabled unless
/// the loaded model implementation proves support. `resolved_from_loaded_model`
/// distinguishes executor truth from catalog-only route planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub backend: BackendKind,
    pub model_variant: Option<ModelVariant>,
    pub mode: ExecutionMode,
    pub prefill: PrefillMode,
    pub incremental_decode: bool,
    pub prefill_batch: NativeBatchMode,
    pub decode_batch: NativeBatchMode,
    pub cache_mode: CacheMode,
    pub cancellation: CancellationGranularity,
    pub concurrency: ConcurrencyClass,
    pub recompute_safe: bool,
    /// The executor can synchronously prove that all model-owned cache state
    /// for an exact session has been released before recomputation or reuse.
    pub cache_release_safe: bool,
    pub prefix_reuse_safe: bool,
    pub max_batch_size: usize,
    pub resolved_from_loaded_model: bool,
    pub compute_dtype: String,
    pub kv_dtype: String,
    pub cache_namespace: Option<String>,
}

impl ExecutionProfile {
    pub fn fail_closed(
        backend: BackendKind,
        model_variant: Option<ModelVariant>,
        mode: ExecutionMode,
    ) -> Self {
        Self {
            backend,
            model_variant,
            mode,
            prefill: PrefillMode::None,
            incremental_decode: false,
            prefill_batch: NativeBatchMode::None,
            decode_batch: NativeBatchMode::None,
            cache_mode: CacheMode::None,
            cancellation: match mode {
                ExecutionMode::Sequence => CancellationGranularity::SequenceStep,
                ExecutionMode::Realtime => CancellationGranularity::RealtimeChunk,
                ExecutionMode::Pipeline => CancellationGranularity::PipelineStage,
                ExecutionMode::Atomic | ExecutionMode::Artifact => {
                    CancellationGranularity::OperationBoundary
                }
            },
            concurrency: ConcurrencyClass::Exclusive,
            recompute_safe: false,
            cache_release_safe: false,
            prefix_reuse_safe: false,
            max_batch_size: 1,
            resolved_from_loaded_model: false,
            compute_dtype: "unknown".to_string(),
            kv_dtype: "none".to_string(),
            cache_namespace: None,
        }
    }

    pub fn capabilities(&self) -> ExecutionCapabilities {
        let native_batch = self.prefill_batch != NativeBatchMode::None
            || self.decode_batch != NativeBatchMode::None;
        ExecutionCapabilities {
            incremental_prefill: self.prefill == PrefillMode::Incremental,
            incremental_decode: self.incremental_decode,
            native_batch,
            mixed_phase_batch: false,
            cancellable_between_steps: !matches!(
                self.cancellation,
                CancellationGranularity::OperationBoundary
            ),
            recompute_safe: self.recompute_safe,
            cache_release_safe: self.cache_release_safe,
            physical_cache: self.cache_mode == CacheMode::ExternalPaged,
            max_batch_size: if native_batch {
                self.max_batch_size.max(1)
            } else {
                1
            },
        }
    }
}

impl Default for ExecutionCapabilities {
    fn default() -> Self {
        Self {
            incremental_prefill: false,
            incremental_decode: false,
            native_batch: false,
            mixed_phase_batch: false,
            cancellable_between_steps: true,
            recompute_safe: false,
            cache_release_safe: false,
            physical_cache: false,
            max_batch_size: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchKey {
    pub backend: BackendKind,
    pub model_variant: Option<ModelVariant>,
    pub task_type: TaskType,
    pub work_kind: String,
    pub compute_dtype: String,
    pub kv_dtype: String,
    pub cache_namespace: String,
    pub adapter_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalOutcome {
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YieldReason {
    QuantumExhausted,
    Backpressure,
    AwaitingInput,
    Preempted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Completed,
    MaxTokens,
    StopToken,
    StopSequence,
    Cancelled,
    TimedOut,
    Rejected,
}

impl FinishReason {
    fn terminal_outcome(self) -> TerminalOutcome {
        match self {
            Self::Completed | Self::MaxTokens | Self::StopToken | Self::StopSequence => {
                TerminalOutcome::Completed
            }
            Self::Cancelled => TerminalOutcome::Cancelled,
            Self::TimedOut => TerminalOutcome::TimedOut,
            Self::Rejected => TerminalOutcome::Rejected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    InvalidOutput,
    Executor,
    Backend,
    ResourceExhausted,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureScope {
    Request,
    Batch,
    Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDisposition {
    Never,
    RetrySameSession,
    Recompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthImpact {
    None,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFailure {
    pub kind: FailureKind,
    pub scope: FailureScope,
    pub retry: RetryDisposition,
    pub health: HealthImpact,
    pub message: String,
}

impl ExecutionFailure {
    pub fn invalid_output(message: impl Into<String>) -> Self {
        Self {
            kind: FailureKind::InvalidOutput,
            scope: FailureScope::Request,
            retry: RetryDisposition::Never,
            health: HealthImpact::None,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionDisposition {
    Progress,
    Yielded(YieldReason),
    Finished(FinishReason),
    Failed(ExecutionFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionState {
    Queued,
    Admitted,
    Prefilling,
    Decoding,
    AtomicRunning,
    PipelineRunning,
    Cancelling,
    PreemptedRecompute,
    Terminal(TerminalOutcome),
}

impl ExecutionState {
    pub fn transition(self, next: Self) -> Result<Self> {
        use ExecutionState::*;
        let legal = matches!(
            (self, next),
            (Queued, Admitted)
                | (Queued, Cancelling)
                | (
                    Queued,
                    Terminal(TerminalOutcome::Rejected | TerminalOutcome::TimedOut)
                )
                | (
                    Admitted,
                    Prefilling | Decoding | AtomicRunning | PipelineRunning | Cancelling
                )
                | (
                    Prefilling,
                    Prefilling | Decoding | Cancelling | PreemptedRecompute
                )
                | (Decoding, Decoding | Cancelling | PreemptedRecompute)
                | (AtomicRunning, Cancelling)
                | (PipelineRunning, PipelineRunning | Cancelling)
                | (PreemptedRecompute, Admitted | Prefilling | Cancelling)
                | (
                    Cancelling,
                    Terminal(TerminalOutcome::Cancelled | TerminalOutcome::TimedOut)
                )
                | (
                    Prefilling | Decoding | AtomicRunning | PipelineRunning,
                    Terminal(_)
                )
        );
        if !legal {
            return Err(Error::InferenceError(format!(
                "illegal execution transition {self:?} -> {next:?}"
            )));
        }
        Ok(next)
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Terminal(_))
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub work: WorkUnit,
    pub batch_key: BatchKey,
    pub batch_mode: NativeBatchMode,
    pub max_batch_size: usize,
    pub estimate: ResourceEstimate,
    pub reservation: Option<ResourceReservation>,
}

#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub input_consumed: usize,
    pub output_produced: usize,
    pub observed_resources: ResourceVector,
    pub dispatch: BatchDispatch,
    pub elapsed: Duration,
    pub safe_point: bool,
    pub disposition: ExecutionDisposition,
    /// Terminal/error flags carried by the executor payload. These are
    /// validated together with `disposition` so the payload cannot claim a
    /// different lifecycle outcome than the execution transaction.
    pub output_finished: bool,
    pub output_has_error: bool,
}

impl ExecutionReport {
    pub fn validate_against(&self, plan: &ExecutionPlan) -> Result<()> {
        if self.plan_id != plan.plan_id || self.session != plan.session {
            return Err(Error::InferenceError(
                "execution report does not match its plan".to_string(),
            ));
        }
        if self.dispatch.width == 0 || self.dispatch.width > plan.max_batch_size.max(1) {
            return Err(Error::InferenceError(
                "execution report has an invalid dispatch width".to_string(),
            ));
        }
        match self.dispatch.kind {
            BatchDispatchKind::Serial if self.dispatch.width != 1 => {
                return Err(Error::InferenceError(
                    "serial executor dispatch must have width one".to_string(),
                ));
            }
            BatchDispatchKind::RequestParallel
                if plan.batch_mode != NativeBatchMode::None || self.dispatch.width < 2 =>
            {
                return Err(Error::InferenceError(
                    "request-parallel dispatch must be a multi-request non-tensor batch"
                        .to_string(),
                ));
            }
            BatchDispatchKind::TensorStatic if plan.batch_mode != NativeBatchMode::Static => {
                return Err(Error::InferenceError(
                    "executor reported an undeclared static tensor batch".to_string(),
                ));
            }
            BatchDispatchKind::TensorContinuous
                if plan.batch_mode != NativeBatchMode::Continuous =>
            {
                return Err(Error::InferenceError(
                    "executor reported an undeclared continuous tensor batch".to_string(),
                ));
            }
            _ => {}
        }
        match plan.work {
            WorkUnit::SequenceStep {
                input,
                max_output_steps,
                ..
            } => {
                if self.input_consumed > input.len() || self.output_produced > max_output_steps {
                    return Err(Error::InferenceError(
                        "executor reported progress beyond the scheduled quantum".to_string(),
                    ));
                }
                if matches!(self.disposition, ExecutionDisposition::Progress)
                    && self.input_consumed == 0
                    && self.output_produced == 0
                {
                    return Err(Error::InferenceError(
                        "executor reported progress without consuming or producing work"
                            .to_string(),
                    ));
                }
            }
            WorkUnit::AtomicJob { .. } => {
                if !matches!(
                    self.disposition,
                    ExecutionDisposition::Finished(_) | ExecutionDisposition::Failed(_)
                ) {
                    return Err(Error::InferenceError(
                        "atomic execution must finish or fail in one transaction".to_string(),
                    ));
                }
            }
            WorkUnit::PipelineStage { .. } => {}
        }
        if matches!(self.disposition, ExecutionDisposition::Yielded(_)) && !self.safe_point {
            return Err(Error::InferenceError(
                "executor may only yield at a declared safe point".to_string(),
            ));
        }
        match &self.disposition {
            ExecutionDisposition::Progress | ExecutionDisposition::Yielded(_) => {
                if self.output_finished || self.output_has_error {
                    return Err(Error::InferenceError(
                        "non-terminal execution returned a terminal or errored payload".to_string(),
                    ));
                }
            }
            ExecutionDisposition::Finished(_) => {
                if !self.output_finished || self.output_has_error {
                    return Err(Error::InferenceError(
                        "finished execution must return a terminal payload without an executor error"
                            .to_string(),
                    ));
                }
            }
            ExecutionDisposition::Failed(failure) => {
                if self.input_consumed != 0 || self.output_produced != 0 {
                    return Err(Error::InferenceError(
                        "failed execution cannot also report committed progress".to_string(),
                    ));
                }
                let terminal = failure.retry == RetryDisposition::Never;
                if self.output_finished != terminal || !self.output_has_error {
                    return Err(Error::InferenceError(if terminal {
                        "non-retryable execution failure must return a terminal errored payload"
                            .to_string()
                    } else {
                        "retryable execution failure must return a non-terminal errored payload"
                            .to_string()
                    }));
                }
                if failure.retry != RetryDisposition::Never && !self.safe_point {
                    return Err(Error::InferenceError(
                        "executor may only retry from a declared safe point".to_string(),
                    ));
                }
                if failure.retry == RetryDisposition::Recompute
                    && !matches!(plan.work, WorkUnit::SequenceStep { .. })
                {
                    return Err(Error::InferenceError(
                        "recompute retry is only valid for sequence execution".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionTracker {
    session: SessionKey,
    state: ExecutionState,
    active_plan: Option<PlanId>,
}

impl ExecutionTracker {
    pub fn new(session: SessionKey) -> Self {
        Self {
            session,
            state: ExecutionState::Queued,
            active_plan: None,
        }
    }

    pub fn session(&self) -> &SessionKey {
        &self.session
    }

    pub fn state(&self) -> ExecutionState {
        self.state
    }

    pub fn active_plan_id(&self) -> Option<PlanId> {
        self.active_plan
    }

    pub fn transition(&mut self, next: ExecutionState) -> Result<()> {
        self.state = self.state.transition(next)?;
        Ok(())
    }

    pub fn begin_plan(&mut self, plan: &ExecutionPlan) -> Result<()> {
        if plan.session != self.session {
            return Err(Error::InferenceError(
                "execution plan belongs to a different request session".to_string(),
            ));
        }
        if self.state.is_terminal() {
            return Err(Error::InferenceError(
                "terminal request cannot begin another execution plan".to_string(),
            ));
        }
        if self.active_plan.is_some() {
            return Err(Error::InferenceError(
                "request already has an active execution plan".to_string(),
            ));
        }
        self.active_plan = Some(plan.plan_id);
        Ok(())
    }

    pub fn commit(&mut self, plan: &ExecutionPlan, report: &ExecutionReport) -> Result<()> {
        report.validate_against(plan)?;
        if self.active_plan != Some(plan.plan_id) {
            return Err(Error::InferenceError(
                "execution report is missing or duplicates a committed plan".to_string(),
            ));
        }
        let next_state = match &report.disposition {
            ExecutionDisposition::Finished(reason) => {
                Some(ExecutionState::Terminal(reason.terminal_outcome()))
            }
            ExecutionDisposition::Failed(failure) if failure.retry == RetryDisposition::Never => {
                Some(ExecutionState::Terminal(TerminalOutcome::Failed))
            }
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::Recompute =>
            {
                Some(ExecutionState::PreemptedRecompute)
            }
            ExecutionDisposition::Progress
            | ExecutionDisposition::Yielded(_)
            | ExecutionDisposition::Failed(_) => None,
        };
        if let Some(next_state) = next_state {
            let validated = self.state.transition(next_state)?;
            self.active_plan = None;
            self.state = validated;
        } else {
            self.active_plan = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_rejects_regressions_and_second_terminal() {
        let state = ExecutionState::Queued
            .transition(ExecutionState::Admitted)
            .unwrap()
            .transition(ExecutionState::Prefilling)
            .unwrap()
            .transition(ExecutionState::Terminal(TerminalOutcome::Completed))
            .unwrap();
        assert!(state.is_terminal());
        assert!(state.transition(ExecutionState::Admitted).is_err());
    }

    #[test]
    fn report_cannot_exceed_sequence_plan() {
        let session = SessionKey::new("request".to_string(), 11);
        let plan = ExecutionPlan {
            plan_id: 7,
            session: session.clone(),
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange::new(4, 8).unwrap(),
                max_output_steps: 1,
            },
            batch_key: BatchKey {
                backend: BackendKind::Cpu,
                model_variant: None,
                task_type: TaskType::Chat,
                work_kind: "prefill".to_string(),
                compute_dtype: "f32".to_string(),
                kv_dtype: "f32".to_string(),
                cache_namespace: "none".to_string(),
                adapter_id: None,
            },
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            estimate: ResourceVector::default(),
            reservation: None,
        };
        let report = ExecutionReport {
            plan_id: 7,
            session,
            input_consumed: 5,
            output_produced: 0,
            observed_resources: ResourceVector::default(),
            dispatch: BatchDispatch::serial(),
            elapsed: Duration::ZERO,
            safe_point: true,
            disposition: ExecutionDisposition::Progress,
            output_finished: false,
            output_has_error: false,
        };
        assert!(report.validate_against(&plan).is_err());
    }

    #[test]
    fn tensor_dispatch_must_match_declared_batch_contract() {
        let mut plan = plan_for(
            SessionKey::new("batch".to_string(), 1),
            WorkUnit::AtomicJob {
                kind: "tts".to_string(),
            },
        );
        plan.batch_mode = NativeBatchMode::Static;
        plan.max_batch_size = 2;
        let mut report = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );
        report.dispatch = BatchDispatch::new(BatchDispatchKind::TensorStatic, 2);
        assert!(report.validate_against(&plan).is_ok());

        report.dispatch = BatchDispatch::new(BatchDispatchKind::TensorContinuous, 2);
        assert!(report.validate_against(&plan).is_err());
        report.dispatch = BatchDispatch::new(BatchDispatchKind::TensorStatic, 3);
        assert!(report.validate_against(&plan).is_err());
    }

    #[test]
    fn request_parallel_dispatch_requires_declared_width_without_tensor_batching() {
        let mut plan = plan_for(
            SessionKey::new("parallel".to_string(), 1),
            WorkUnit::AtomicJob {
                kind: "chat".to_string(),
            },
        );
        plan.max_batch_size = 4;
        let mut report = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );

        report.dispatch = BatchDispatch::new(BatchDispatchKind::RequestParallel, 4);
        assert!(report.validate_against(&plan).is_ok());

        report.dispatch = BatchDispatch::new(BatchDispatchKind::RequestParallel, 1);
        assert!(report.validate_against(&plan).is_err());
        report.dispatch = BatchDispatch::new(BatchDispatchKind::Serial, 2);
        assert!(report.validate_against(&plan).is_err());

        plan.batch_mode = NativeBatchMode::Static;
        report.dispatch = BatchDispatch::new(BatchDispatchKind::RequestParallel, 2);
        assert!(report.validate_against(&plan).is_err());
    }

    fn plan_for(session: SessionKey, work: WorkUnit) -> ExecutionPlan {
        ExecutionPlan {
            plan_id: 7,
            session,
            work,
            batch_key: BatchKey {
                backend: BackendKind::Cpu,
                model_variant: None,
                task_type: TaskType::Chat,
                work_kind: "test".to_string(),
                compute_dtype: "f32".to_string(),
                kv_dtype: "f32".to_string(),
                cache_namespace: "none".to_string(),
                adapter_id: None,
            },
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            estimate: ResourceVector::zero(),
            reservation: None,
        }
    }

    fn report_for(plan: &ExecutionPlan, disposition: ExecutionDisposition) -> ExecutionReport {
        let (output_finished, output_has_error) = match &disposition {
            ExecutionDisposition::Progress | ExecutionDisposition::Yielded(_) => (false, false),
            ExecutionDisposition::Finished(_) => (true, false),
            ExecutionDisposition::Failed(failure) => {
                (failure.retry == RetryDisposition::Never, true)
            }
        };
        ExecutionReport {
            plan_id: plan.plan_id,
            session: plan.session.clone(),
            input_consumed: 0,
            output_produced: 0,
            observed_resources: ResourceVector::zero(),
            dispatch: BatchDispatch::serial(),
            elapsed: Duration::ZERO,
            safe_point: true,
            disposition,
            output_finished,
            output_has_error,
        }
    }

    #[test]
    fn reports_are_fenced_by_session_epoch_and_plan_id() {
        let session = SessionKey::new("same-id".to_string(), 3);
        let plan = plan_for(
            session,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
            },
        );
        let mut wrong_epoch = report_for(
            &plan,
            ExecutionDisposition::Yielded(YieldReason::AwaitingInput),
        );
        wrong_epoch.session.epoch += 1;
        assert!(wrong_epoch.validate_against(&plan).is_err());

        let mut wrong_plan = report_for(
            &plan,
            ExecutionDisposition::Yielded(YieldReason::AwaitingInput),
        );
        wrong_plan.plan_id += 1;
        assert!(wrong_plan.validate_against(&plan).is_err());
    }

    #[test]
    fn sequence_progress_and_yields_have_explicit_semantics() {
        let plan = plan_for(
            SessionKey::new("request".to_string(), 1),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
            },
        );
        let no_progress = report_for(&plan, ExecutionDisposition::Progress);
        assert!(no_progress.validate_against(&plan).is_err());

        let mut yielded = report_for(
            &plan,
            ExecutionDisposition::Yielded(YieldReason::AwaitingInput),
        );
        assert!(yielded.validate_against(&plan).is_ok());
        yielded.safe_point = false;
        assert!(yielded.validate_against(&plan).is_err());
    }

    #[test]
    fn atomic_work_must_finish_or_fail() {
        let plan = plan_for(
            SessionKey::new("request".to_string(), 1),
            WorkUnit::AtomicJob {
                kind: "chat".to_string(),
            },
        );
        assert!(report_for(&plan, ExecutionDisposition::Progress)
            .validate_against(&plan)
            .is_err());
        assert!(report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed)
        )
        .validate_against(&plan)
        .is_ok());
    }

    #[test]
    fn tracker_preserves_active_plan_after_invalid_or_duplicate_operations() {
        let session = SessionKey::new("request".to_string(), 1);
        let plan = plan_for(
            session.clone(),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
            },
        );
        let mut tracker = ExecutionTracker::new(session);
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::Decoding).unwrap();
        tracker.begin_plan(&plan).unwrap();
        assert!(tracker.begin_plan(&plan).is_err());
        assert_eq!(tracker.active_plan_id(), Some(plan.plan_id));

        let mut wrong = report_for(
            &plan,
            ExecutionDisposition::Yielded(YieldReason::AwaitingInput),
        );
        wrong.plan_id += 1;
        assert!(tracker.commit(&plan, &wrong).is_err());
        assert_eq!(tracker.active_plan_id(), Some(plan.plan_id));

        let valid = report_for(
            &plan,
            ExecutionDisposition::Yielded(YieldReason::AwaitingInput),
        );
        tracker.commit(&plan, &valid).unwrap();
        assert!(tracker.commit(&plan, &valid).is_err());
    }

    #[test]
    fn retry_policy_controls_whether_failure_terminalizes_the_session() {
        let session = SessionKey::new("request".to_string(), 1);
        let plan = plan_for(
            session.clone(),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
            },
        );
        let mut tracker = ExecutionTracker::new(session);
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::Decoding).unwrap();
        tracker.begin_plan(&plan).unwrap();
        let retryable = ExecutionFailure {
            kind: FailureKind::Backend,
            scope: FailureScope::Request,
            retry: RetryDisposition::RetrySameSession,
            health: HealthImpact::Degraded,
            message: "transient".to_string(),
        };
        tracker
            .commit(
                &plan,
                &report_for(&plan, ExecutionDisposition::Failed(retryable)),
            )
            .unwrap();
        assert_eq!(tracker.state(), ExecutionState::Decoding);

        tracker.begin_plan(&plan).unwrap();
        tracker
            .commit(
                &plan,
                &report_for(
                    &plan,
                    ExecutionDisposition::Failed(ExecutionFailure::invalid_output("bad")),
                ),
            )
            .unwrap();
        assert_eq!(
            tracker.state(),
            ExecutionState::Terminal(TerminalOutcome::Failed)
        );
    }

    #[test]
    fn disposition_and_payload_terminal_state_cannot_disagree() {
        let plan = plan_for(
            SessionKey::new("request".to_string(), 1),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
            },
        );

        let mut completed = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );
        completed.output_finished = false;
        assert!(completed.validate_against(&plan).is_err());

        let retry = ExecutionFailure {
            kind: FailureKind::Backend,
            scope: FailureScope::Request,
            retry: RetryDisposition::RetrySameSession,
            health: HealthImpact::Degraded,
            message: "transient".to_string(),
        };
        let mut retryable = report_for(&plan, ExecutionDisposition::Failed(retry));
        retryable.output_finished = true;
        assert!(retryable.validate_against(&plan).is_err());
    }

    #[test]
    fn recompute_failure_moves_tracker_to_recompute_state() {
        let session = SessionKey::new("request".to_string(), 1);
        let plan = plan_for(
            session.clone(),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
            },
        );
        let mut tracker = ExecutionTracker::new(session);
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::Decoding).unwrap();
        tracker.begin_plan(&plan).unwrap();

        let recompute = ExecutionFailure {
            kind: FailureKind::Backend,
            scope: FailureScope::Request,
            retry: RetryDisposition::Recompute,
            health: HealthImpact::Degraded,
            message: "cache invalidated".to_string(),
        };
        tracker
            .commit(
                &plan,
                &report_for(&plan, ExecutionDisposition::Failed(recompute)),
            )
            .unwrap();

        assert_eq!(tracker.state(), ExecutionState::PreemptedRecompute);
        assert_eq!(tracker.active_plan_id(), None);
    }

    #[test]
    fn execution_profiles_fail_closed_until_features_are_proven() {
        let profile = ExecutionProfile::fail_closed(
            BackendKind::Cuda,
            Some(ModelVariant::Qwen306B),
            ExecutionMode::Atomic,
        );
        let capabilities = profile.capabilities();

        assert_eq!(profile.prefill, PrefillMode::None);
        assert_eq!(profile.cache_mode, CacheMode::None);
        assert_eq!(profile.concurrency, ConcurrencyClass::Exclusive);
        assert!(!profile.resolved_from_loaded_model);
        assert!(!capabilities.incremental_prefill);
        assert!(!capabilities.incremental_decode);
        assert!(!capabilities.native_batch);
        assert!(!capabilities.cancellable_between_steps);
        assert!(!capabilities.recompute_safe);
        assert!(!capabilities.physical_cache);
        assert_eq!(capabilities.max_batch_size, 1);
    }

    #[test]
    fn profile_capabilities_only_expose_declared_features() {
        let mut profile = ExecutionProfile::fail_closed(
            BackendKind::Metal,
            Some(ModelVariant::Qwen306B),
            ExecutionMode::Sequence,
        );
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.decode_batch = NativeBatchMode::Static;
        profile.cache_mode = CacheMode::OpaqueModelOwned;
        profile.max_batch_size = 4;

        let capabilities = profile.capabilities();
        assert!(!capabilities.incremental_prefill);
        assert!(capabilities.incremental_decode);
        assert!(capabilities.native_batch);
        assert!(capabilities.cancellable_between_steps);
        assert!(!capabilities.physical_cache);
        assert_eq!(capabilities.max_batch_size, 4);
    }
}

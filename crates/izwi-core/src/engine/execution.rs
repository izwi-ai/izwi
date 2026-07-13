//! Authoritative execution plans, reports, capabilities, and lifecycle states.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::resources::{ResourceEstimate, ResourceReservation, ResourceVector};
use super::{RequestId, TaskType};

pub type PlanId = u64;

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
                | (Decoding, Decoding | Cancelling)
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
    pub request_id: RequestId,
    pub work: WorkUnit,
    pub batch_key: BatchKey,
    pub estimate: ResourceEstimate,
    pub reservation: Option<ResourceReservation>,
}

#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub plan_id: PlanId,
    pub request_id: RequestId,
    pub input_consumed: usize,
    pub output_produced: usize,
    pub observed_resources: ResourceVector,
    pub elapsed: Duration,
    pub safe_point: bool,
    pub terminal: Option<TerminalOutcome>,
}

impl ExecutionReport {
    pub fn validate_against(&self, plan: &ExecutionPlan) -> Result<()> {
        if self.plan_id != plan.plan_id || self.request_id != plan.request_id {
            return Err(Error::InferenceError(
                "execution report does not match its plan".to_string(),
            ));
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
            }
            WorkUnit::AtomicJob { .. } | WorkUnit::PipelineStage { .. } => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionTracker {
    state: ExecutionState,
    active_plan: Option<PlanId>,
}

impl Default for ExecutionTracker {
    fn default() -> Self {
        Self {
            state: ExecutionState::Queued,
            active_plan: None,
        }
    }
}

impl ExecutionTracker {
    pub fn state(&self) -> ExecutionState {
        self.state
    }

    pub fn transition(&mut self, next: ExecutionState) -> Result<()> {
        self.state = self.state.transition(next)?;
        Ok(())
    }

    pub fn begin_plan(&mut self, plan: &ExecutionPlan) -> Result<()> {
        if self.active_plan.replace(plan.plan_id).is_some() {
            return Err(Error::InferenceError(
                "request already has an active execution plan".to_string(),
            ));
        }
        Ok(())
    }

    pub fn commit(&mut self, plan: &ExecutionPlan, report: &ExecutionReport) -> Result<()> {
        report.validate_against(plan)?;
        if self.active_plan != Some(plan.plan_id) {
            return Err(Error::InferenceError(
                "execution report is missing or duplicates a committed plan".to_string(),
            ));
        }
        self.active_plan = None;
        if let Some(outcome) = report.terminal {
            self.transition(ExecutionState::Terminal(outcome))?;
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
        let plan = ExecutionPlan {
            plan_id: 7,
            request_id: "request".to_string(),
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
            estimate: ResourceVector::default(),
            reservation: None,
        };
        let report = ExecutionReport {
            plan_id: 7,
            request_id: "request".to_string(),
            input_consumed: 5,
            output_produced: 0,
            observed_resources: ResourceVector::default(),
            elapsed: Duration::ZERO,
            safe_point: true,
            terminal: None,
        };
        assert!(report.validate_against(&plan).is_err());
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

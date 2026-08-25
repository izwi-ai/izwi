//! Authoritative execution plans, reports, capabilities, and lifecycle states.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::backends::kv::KvWriteBatchCompletion;
use crate::backends::state::{TensorStateBatchCompletion, TensorStateSelection};
use crate::backends::BackendKind;
use crate::config::PhysicalInFlightLimit;
use crate::engine::cache::coordinator::GroupBlockTable;
use crate::error::{Error, Result};
pub use crate::kv::v2::{StateClock, StateGroupId};
use crate::kv::{CacheBlockRef, CacheDomainId, KvArenaId, KvSlotRef};
use crate::model::ModelVariant;

use super::resources::{ResourceEstimate, ResourceVector};
use super::{RequestId, SequenceId, TaskType};

pub type PlanId = u64;
pub type SessionEpoch = SequenceId;

macro_rules! execution_id {
    ($name:ident, $value:ty) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name($value);

        impl $name {
            pub const fn new(value: $value) -> Self {
                Self(value)
            }

            pub const fn get(self) -> $value {
                self.0
            }
        }
    };
}

execution_id!(ExecutionGroupId, u64);
execution_id!(ModelInstanceId, u64);
execution_id!(AdapterInstanceId, u64);
execution_id!(AdapterAbiRevision, u32);
execution_id!(StageId, u32);
execution_id!(BatchId, u64);

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

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Load-authored authorization for one independently clocked retained-state
/// group. A selection permits a stage to advance the group; it does not by
/// itself require every row or quantum to do so.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClockedStateSelection {
    group: StateGroupId,
    clock: StateClock,
}

impl ClockedStateSelection {
    pub fn new(group: StateGroupId, clock: StateClock) -> Result<Self> {
        if group.get() == 0 {
            return Err(Error::InvalidInput(
                "clocked retained-state selection has a zero group id".into(),
            ));
        }
        if matches!(&clock, StateClock::Custom(name) if name.trim().is_empty()) {
            return Err(Error::InvalidInput(
                "custom retained-state clock name cannot be empty".into(),
            ));
        }
        Ok(Self { group, clock })
    }

    pub const fn group(&self) -> StateGroupId {
        self.group
    }

    pub const fn clock(&self) -> &StateClock {
        &self.clock
    }
}

/// Exact input interval by which one retained-state group advances in a
/// sequence quantum. Spans are sealed by Core after exact stage and physical
/// state-plan authentication; executors may consume but never invent them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClockedStateSpan {
    group: StateGroupId,
    clock: StateClock,
    input: InputRange,
}

impl ClockedStateSpan {
    pub fn new(group: StateGroupId, clock: StateClock, input: InputRange) -> Result<Self> {
        if group.get() == 0 {
            return Err(Error::InvalidInput(
                "clocked retained-state span has a zero group id".into(),
            ));
        }
        if input.is_empty() {
            return Err(Error::InvalidInput(
                "clocked retained-state span cannot be empty".into(),
            ));
        }
        Ok(Self {
            group,
            clock,
            input,
        })
    }

    pub const fn group(&self) -> StateGroupId {
        self.group
    }

    pub const fn clock(&self) -> &StateClock {
        &self.clock
    }

    pub const fn input(&self) -> InputRange {
        self.input
    }
}

/// Immutable request-owned mapping from a primary sequence interval to an
/// independently clocked retained-state interval. Preparation authors these
/// projections; Core intersects and seals them for each scheduled quantum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClockedStateProjection {
    primary: InputRange,
    selection: ClockedStateSelection,
    auxiliary: InputRange,
    scale: usize,
}

impl ClockedStateProjection {
    pub(crate) fn new(
        primary: InputRange,
        selection: ClockedStateSelection,
        auxiliary: InputRange,
    ) -> Result<Self> {
        if primary.is_empty() || auxiliary.is_empty() {
            return Err(Error::InvalidInput(
                "clocked retained-state projection ranges cannot be empty".into(),
            ));
        }
        if auxiliary.len() % primary.len() != 0 {
            return Err(Error::InvalidInput(
                "clocked retained-state projection has no exact integral scale".into(),
            ));
        }
        let scale = auxiliary.len() / primary.len();
        if scale == 0 {
            return Err(Error::InvalidInput(
                "clocked retained-state projection has a zero scale".into(),
            ));
        }
        Ok(Self {
            primary,
            selection,
            auxiliary,
            scale,
        })
    }

    pub(crate) const fn selection(&self) -> &ClockedStateSelection {
        &self.selection
    }

    pub(crate) const fn primary(&self) -> InputRange {
        self.primary
    }

    pub(crate) const fn auxiliary(&self) -> InputRange {
        self.auxiliary
    }

    pub(crate) const fn scale(&self) -> usize {
        self.scale
    }

    pub(crate) fn project(&self, scheduled: InputRange) -> Result<Option<ClockedStateSpan>> {
        let start = scheduled.start.max(self.primary.start);
        let end = scheduled.end.min(self.primary.end);
        if end <= start {
            return Ok(None);
        }
        let mapped_start = start
            .checked_sub(self.primary.start)
            .and_then(|offset| offset.checked_mul(self.scale))
            .and_then(|offset| self.auxiliary.start.checked_add(offset))
            .ok_or_else(|| Error::InvalidInput("clocked state projection start overflow".into()))?;
        let mapped_end = end
            .checked_sub(self.primary.start)
            .and_then(|offset| offset.checked_mul(self.scale))
            .and_then(|offset| self.auxiliary.start.checked_add(offset))
            .ok_or_else(|| Error::InvalidInput("clocked state projection end overflow".into()))?;
        if mapped_end > self.auxiliary.end {
            return Err(Error::InvalidInput(
                "clocked state projection exceeds its authenticated auxiliary interval".into(),
            ));
        }
        ClockedStateSpan::new(
            self.selection.group(),
            self.selection.clock().clone(),
            InputRange::new(mapped_start, mapped_end)?,
        )
        .map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequencePhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkUnit {
    /// A model stage that must complete before the request's execution shape
    /// and persistent sequence state can be admitted. Preparation is a
    /// distinct transaction: it must not be represented as decoder prefill.
    PreSequencePreparation {
        kind: String,
    },
    SequenceStep {
        phase: SequencePhase,
        input: InputRange,
        max_output_steps: usize,
        /// Core-sealed retained tensor/static-state work distinct from the
        /// primary decoder-token interval.
        /// `None` preserves legacy decoder-coupled retained state. `Some([])`
        /// is an authenticated explicit selection of no auxiliary group.
        auxiliary_state: Option<Arc<[ClockedStateSpan]>>,
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

/// Where one model stage is executed. Host stages remain part of the same
/// logical workflow but do not consume a device execution-group permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDomain {
    Host,
    ExecutionGroup,
}

/// How a stage makes observable progress. Continuous batch membership is only
/// valid for stages that expose repeatable or input-driven safe points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageProgressKind {
    Atomic,
    Iterative,
    InputDriven,
}

/// Model-owned routing from a scheduler work quantum to one execution stage.
/// Exact selectors take precedence over a single compatibility fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageWorkSelector {
    Any,
    PreSequencePreparation,
    SequencePrefill,
    SequenceDecode,
    Atomic,
    Pipeline { ordinal: Option<usize> },
}

impl StageWorkSelector {
    fn matches(self, work: &WorkUnit) -> bool {
        match (self, work) {
            (Self::Any, _) => true,
            (Self::PreSequencePreparation, WorkUnit::PreSequencePreparation { .. }) => true,
            (
                Self::SequencePrefill,
                WorkUnit::SequenceStep {
                    phase: SequencePhase::Prefill,
                    ..
                },
            )
            | (
                Self::SequenceDecode,
                WorkUnit::SequenceStep {
                    phase: SequencePhase::Decode,
                    ..
                },
            )
            | (Self::Atomic, WorkUnit::AtomicJob { .. }) => true,
            (
                Self::Pipeline { ordinal },
                WorkUnit::PipelineStage {
                    ordinal: work_ordinal,
                    ..
                },
            ) => ordinal.is_none_or(|ordinal| ordinal == *work_ordinal),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageShapePolicy {
    /// Rows execute independently and therefore have no shared tensor shape.
    Independent,
    Exact,
    Bucketed,
    Padded,
    Ragged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MembershipSafePoint {
    OperationBoundary,
    QuantumBoundary,
    InputBoundary,
    PipelineBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputVisibility {
    /// Executor output remains private until the physical report and the
    /// corresponding model/scheduler state transition commit together.
    AfterQuantumCommit,
    /// A non-tensor stage may commit fenced, non-terminal progress records
    /// while its physical operation is still running. The authoritative final
    /// marker remains gated by the normal physical report and state commit.
    IncrementalCommitted,
}

/// Model-owned description of one execution stage. The engine treats `id` as
/// opaque and never branches on cache, transducer, diffusion, or codec types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageDescriptor {
    pub id: StageId,
    pub name: String,
    pub selector: StageWorkSelector,
    pub domain: ExecutionDomain,
    pub progress: StageProgressKind,
    pub concurrency: ConcurrencyClass,
    /// Load-sealed certification for overlap between distinct physical calls.
    /// This remains independent from row formation within one physical batch.
    #[serde(default)]
    pub physical_launch_policy: PhysicalLaunchPolicy,
    pub batch_mode: NativeBatchMode,
    pub max_batch_size: usize,
    pub max_work_units: u64,
    pub workspace_base_bytes: u64,
    pub workspace_per_row_bytes: u64,
    pub workspace_per_work_unit_bytes: u64,
    pub max_workspace_bytes: u64,
    pub max_padding_basis_points: u16,
    pub max_formation_delay: Duration,
    pub shape_policy: StageShapePolicy,
    pub membership_safe_point: MembershipSafePoint,
    pub output_visibility: OutputVisibility,
    /// Canonical load-authored authorization for independently clocked
    /// retained groups this stage may advance.
    #[serde(default)]
    /// `None` preserves the legacy contract in which every retained group is
    /// coupled to the primary decoder cursor. `Some([])` explicitly selects no
    /// auxiliary group; a non-empty value authorizes exactly those groups.
    pub retained_state_selections: Option<Vec<ClockedStateSelection>>,
}

impl StageDescriptor {
    /// Conservative bridge for existing executors. Callers choose the phase's
    /// declared batch mode explicitly because one legacy profile can describe
    /// different prefill and decode behavior.
    pub fn from_execution_profile(
        id: StageId,
        name: impl Into<String>,
        profile: &ExecutionProfile,
        batch_mode: NativeBatchMode,
    ) -> Self {
        let progress = match profile.mode {
            ExecutionMode::Sequence => StageProgressKind::Iterative,
            ExecutionMode::Realtime => StageProgressKind::InputDriven,
            ExecutionMode::Atomic | ExecutionMode::Pipeline | ExecutionMode::Artifact => {
                StageProgressKind::Atomic
            }
        };
        let membership_safe_point = match profile.cancellation {
            CancellationGranularity::OperationBoundary => MembershipSafePoint::OperationBoundary,
            CancellationGranularity::SequenceStep => MembershipSafePoint::QuantumBoundary,
            CancellationGranularity::RealtimeChunk => MembershipSafePoint::InputBoundary,
            CancellationGranularity::PipelineStage => MembershipSafePoint::PipelineBoundary,
        };
        let concurrency = if batch_mode == NativeBatchMode::None {
            profile.concurrency
        } else {
            ConcurrencyClass::Batchable
        };
        let shape_policy = match (batch_mode, concurrency) {
            (NativeBatchMode::None, ConcurrencyClass::Batchable) => StageShapePolicy::Independent,
            (NativeBatchMode::None, ConcurrencyClass::Exclusive) => StageShapePolicy::Exact,
            (NativeBatchMode::Static, _) => StageShapePolicy::Padded,
            (NativeBatchMode::Continuous, _) => StageShapePolicy::Ragged,
        };
        Self {
            id,
            name: name.into(),
            selector: StageWorkSelector::Any,
            domain: if profile.mode == ExecutionMode::Artifact {
                ExecutionDomain::Host
            } else {
                ExecutionDomain::ExecutionGroup
            },
            progress,
            concurrency,
            physical_launch_policy: profile.effective_physical_launch_policy(),
            batch_mode,
            max_batch_size: profile.max_batch_size.max(1),
            max_work_units: u64::MAX,
            workspace_base_bytes: 0,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes: 0,
            max_padding_basis_points: if shape_policy == StageShapePolicy::Padded {
                10_000
            } else {
                0
            },
            max_formation_delay: Duration::ZERO,
            shape_policy,
            membership_safe_point,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::InvalidInput(
                "execution stage name cannot be empty".to_string(),
            ));
        }
        let mut previous_group = None;
        for selection in self
            .retained_state_selections
            .as_deref()
            .unwrap_or_default()
        {
            if selection.group().get() == 0
                || previous_group.is_some_and(|previous| previous >= selection.group().get())
            {
                return Err(Error::InvalidInput(
                    "execution stage retained-state selections must have nonzero, strictly increasing group ids"
                        .into(),
                ));
            }
            previous_group = Some(selection.group().get());
        }
        if self.max_batch_size == 0 || self.max_work_units == 0 {
            return Err(Error::InvalidInput(
                "execution stage budgets must be greater than zero".to_string(),
            ));
        }
        if self.max_padding_basis_points > 10_000 {
            return Err(Error::InvalidInput(
                "execution stage padding budget cannot exceed 100 percent".to_string(),
            ));
        }
        if self.workspace_base_bytes > self.max_workspace_bytes
            || self.workspace_per_row_bytes > self.max_workspace_bytes
            || self.workspace_per_work_unit_bytes > self.max_workspace_bytes
        {
            return Err(Error::InvalidInput(
                "execution stage workspace estimate exceeds its maximum".to_string(),
            ));
        }
        if self.concurrency == ConcurrencyClass::Exclusive && self.max_batch_size != 1 {
            return Err(Error::InvalidInput(
                "exclusive execution stages must have width one".to_string(),
            ));
        }
        if self.batch_mode != NativeBatchMode::None
            && self.concurrency != ConcurrencyClass::Batchable
        {
            return Err(Error::InvalidInput(
                "native tensor stages must be batchable".to_string(),
            ));
        }
        if self.batch_mode == NativeBatchMode::None
            && self.concurrency == ConcurrencyClass::Batchable
            && self.shape_policy != StageShapePolicy::Independent
        {
            return Err(Error::InvalidInput(
                "request-parallel stages must use independent row shapes".to_string(),
            ));
        }
        if matches!(
            self.physical_launch_policy,
            PhysicalLaunchPolicy::Concurrent { .. }
        ) && (self.domain != ExecutionDomain::ExecutionGroup
            || self.batch_mode != NativeBatchMode::None
            || self.concurrency != ConcurrencyClass::Batchable
            || self.shape_policy != StageShapePolicy::Independent)
        {
            return Err(Error::InvalidInput(
                "concurrent physical launches require independent scalar execution-group rows"
                    .to_string(),
            ));
        }
        if self.shape_policy != StageShapePolicy::Padded && self.max_padding_basis_points != 0 {
            return Err(Error::InvalidInput(
                "only padded execution stages may declare padding overhead".to_string(),
            ));
        }
        if self.batch_mode == NativeBatchMode::Continuous
            && (self.progress == StageProgressKind::Atomic
                || self.membership_safe_point == MembershipSafePoint::OperationBoundary)
        {
            return Err(Error::InvalidInput(
                "continuous batching requires a repeatable membership safe point".to_string(),
            ));
        }
        if self.output_visibility == OutputVisibility::IncrementalCommitted
            && self.batch_mode != NativeBatchMode::None
        {
            return Err(Error::InvalidInput(
                "native tensor stages cannot publish in-flight output checkpoints".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdapterBindingKey {
    pub execution_group_id: ExecutionGroupId,
    pub model_instance_id: ModelInstanceId,
    pub adapter_instance_id: AdapterInstanceId,
    pub adapter_abi_revision: AdapterAbiRevision,
    pub capability_id: String,
    pub stage_id: StageId,
}

/// Exact loaded adapter selected before scheduler admission. The binding is
/// immutable for one request incarnation and survives until terminal cleanup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionAdapterBinding {
    pub execution_group_id: ExecutionGroupId,
    pub model_instance_id: ModelInstanceId,
    pub adapter_instance_id: AdapterInstanceId,
    pub adapter_abi_revision: AdapterAbiRevision,
    pub model_variant: ModelVariant,
    pub capability_id: String,
    pub stages: Arc<[StageDescriptor]>,
}

impl ExecutionAdapterBinding {
    pub fn validate(&self) -> Result<()> {
        if self.execution_group_id.get() == 0
            || self.model_instance_id.get() == 0
            || self.adapter_instance_id.get() == 0
            || self.adapter_abi_revision.get() == 0
        {
            return Err(Error::InvalidInput(
                "execution adapter binding contains a zero lifecycle identity".to_string(),
            ));
        }
        if self.capability_id.trim().is_empty() {
            return Err(Error::InvalidInput(
                "execution adapter binding has an empty capability identity".to_string(),
            ));
        }
        if self.stages.is_empty() {
            return Err(Error::InvalidInput(
                "execution adapter binding has no stages".to_string(),
            ));
        }
        let mut stage_ids = HashSet::with_capacity(self.stages.len());
        for stage in self.stages.iter() {
            stage.validate()?;
            if !stage_ids.insert(stage.id) {
                return Err(Error::InvalidInput(
                    "execution adapter binding contains a duplicate stage identity".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn primary_stage(&self) -> &StageDescriptor {
        &self.stages[0]
    }

    pub fn stage_for_work(&self, work: &WorkUnit) -> Result<&StageDescriptor> {
        let mut exact = self.stages.iter().filter(|stage| {
            stage.selector != StageWorkSelector::Any && stage.selector.matches(work)
        });
        if let Some(stage) = exact.next() {
            if exact.next().is_some() {
                return Err(Error::InvalidInput(
                    "execution adapter has ambiguous exact stage selectors".to_string(),
                ));
            }
            return Ok(stage);
        }

        let mut fallback = self
            .stages
            .iter()
            .filter(|stage| stage.selector == StageWorkSelector::Any);
        let stage = fallback.next().ok_or_else(|| {
            Error::InvalidInput("execution adapter has no stage for scheduled work".to_string())
        })?;
        if fallback.next().is_some() {
            return Err(Error::InvalidInput(
                "execution adapter has multiple fallback stages".to_string(),
            ));
        }
        Ok(stage)
    }

    pub fn key_for_stage(&self, stage_id: StageId) -> Result<AdapterBindingKey> {
        if !self.stages.iter().any(|stage| stage.id == stage_id) {
            return Err(Error::InvalidInput(
                "execution adapter binding does not contain the requested stage".to_string(),
            ));
        }
        Ok(AdapterBindingKey {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            adapter_instance_id: self.adapter_instance_id,
            adapter_abi_revision: self.adapter_abi_revision,
            capability_id: self.capability_id.clone(),
            stage_id,
        })
    }
}

/// Backend-neutral cost of one safe execution quantum. Logical units may be
/// tokens, audio frames, samples, codec frames, or another adapter-defined unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkCost {
    pub logical_units: u64,
    pub tensor_elements: u64,
    pub workspace: ResourceVector,
}

impl WorkCost {
    pub const fn new(logical_units: u64, tensor_elements: u64, workspace_bytes: u64) -> Self {
        Self {
            logical_units,
            tensor_elements,
            workspace: ResourceVector::temporary_workspace(workspace_bytes),
        }
    }

    pub const fn with_workspace(
        logical_units: u64,
        tensor_elements: u64,
        workspace: ResourceVector,
    ) -> Self {
        Self {
            logical_units,
            tensor_elements,
            workspace,
        }
    }

    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            logical_units: self.logical_units.checked_add(other.logical_units)?,
            tensor_elements: self.tensor_elements.checked_add(other.tensor_elements)?,
            workspace: self.workspace.checked_add(other.workspace).ok()?,
        })
    }
}

impl Default for WorkCost {
    fn default() -> Self {
        Self::with_workspace(0, 0, ResourceVector::zero())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchBudget {
    pub max_rows: usize,
    pub max_logical_units: u64,
    pub max_tensor_elements: u64,
    pub max_workspace_bytes: u64,
    /// Maximum padded work as basis points of useful work. `10_000` permits
    /// padding equal to the useful tensor work.
    pub max_padding_basis_points: u16,
    pub max_formation_delay: Duration,
}

impl BatchBudget {
    pub const fn width_one() -> Self {
        Self {
            max_rows: 1,
            max_logical_units: u64::MAX,
            max_tensor_elements: u64::MAX,
            max_workspace_bytes: u64::MAX,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.max_rows == 0 || self.max_logical_units == 0 || self.max_tensor_elements == 0 {
            return Err(Error::InvalidInput(
                "physical batch budgets must be greater than zero".to_string(),
            ));
        }
        if self.max_padding_basis_points > 10_000 {
            return Err(Error::InvalidInput(
                "physical batch padding budget cannot exceed 100 percent".to_string(),
            ));
        }
        Ok(())
    }

    pub fn admits(self, current_rows: usize, current: WorkCost, next: WorkCost) -> bool {
        let Some(rows) = current_rows.checked_add(1) else {
            return false;
        };
        let Some(total) = current.checked_add(next) else {
            return false;
        };
        rows <= self.max_rows
            && total.logical_units <= self.max_logical_units
            && total.tensor_elements <= self.max_tensor_elements
            && total
                .workspace
                .workspace_bytes()
                .is_ok_and(|bytes| bytes <= self.max_workspace_bytes)
    }
}

/// Observed dispatch mechanism for one executor report. Request-parallel work
/// is intentionally distinct from a model tensor batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchDispatchKind {
    #[default]
    Serial,
    NotDispatched,
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

    pub const fn not_dispatched(width: usize) -> Self {
        Self {
            kind: BatchDispatchKind::NotDispatched,
            width,
        }
    }
}

impl Default for BatchDispatch {
    fn default() -> Self {
        Self::serial()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeadlinePhase {
    SchedulerQueue,
    DispatchWait,
    ModelExecution,
    StreamDelivery,
    TerminalDelivery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchState {
    NotStarted,
    Started,
    ProducedOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureOrigin {
    AdapterPlanning,
    DispatchCoordination,
    WorkspaceAdmission,
    ExecutorValidation,
    Model,
    StreamDelivery,
    StateCommit,
    Cleanup,
    Panic,
}

/// Bounded execution provenance carried from physical dispatch through the
/// terminal API result. Detailed error text remains separate and unlabelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeProvenance {
    pub dispatch_state: DispatchState,
    pub failure_origin: Option<FailureOrigin>,
    pub deadline_phase: Option<DeadlinePhase>,
}

impl OutcomeProvenance {
    pub const fn not_started() -> Self {
        Self {
            dispatch_state: DispatchState::NotStarted,
            failure_origin: None,
            deadline_phase: None,
        }
    }

    pub const fn produced_output() -> Self {
        Self {
            dispatch_state: DispatchState::ProducedOutput,
            failure_origin: None,
            deadline_phase: None,
        }
    }

    pub const fn started() -> Self {
        Self {
            dispatch_state: DispatchState::Started,
            failure_origin: None,
            deadline_phase: None,
        }
    }

    pub const fn failure(origin: FailureOrigin, dispatch_state: DispatchState) -> Self {
        Self {
            dispatch_state,
            failure_origin: Some(origin),
            deadline_phase: None,
        }
    }

    pub const fn deadline(phase: DeadlinePhase, dispatch_state: DispatchState) -> Self {
        Self {
            dispatch_state,
            failure_origin: None,
            deadline_phase: Some(phase),
        }
    }
}

impl Default for OutcomeProvenance {
    fn default() -> Self {
        Self::produced_output()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    None,
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
    /// One logical row per physical-batch envelope.
    Exclusive,
    /// Multiple compatible rows may share one physical-batch envelope, either
    /// as a tensor invocation or as independent compatibility rows. This does
    /// not certify overlap between distinct physical model calls.
    Batchable,
}

/// Backend/model certification for overlapping distinct physical launches.
///
/// This is intentionally separate from [`ConcurrencyClass`], which describes
/// row formation within one physical invocation. Policies fail closed to
/// execution-group serialization until a loaded backend/model contract
/// explicitly certifies a narrower scope or concurrent model calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhysicalLaunchPolicy {
    #[default]
    ExecutionGroupExclusive,
    ModelExclusive,
    Concurrent {
        /// Per-model ceiling, independently clamped by the engine-wide
        /// physical in-flight limit.
        max_in_flight_per_model: PhysicalInFlightLimit,
    },
}

impl PhysicalLaunchPolicy {
    pub fn concurrent(max_in_flight_per_model: usize) -> Result<Self> {
        Ok(Self::Concurrent {
            max_in_flight_per_model: PhysicalInFlightLimit::new(max_in_flight_per_model).map_err(
                |_| {
                    Error::InvalidInput(
                        "concurrent physical launch policy requires a non-zero per-model limit"
                            .to_string(),
                    )
                },
            )?,
        })
    }

    /// Effective same-model overlap after applying the engine-wide physical
    /// launch limit. Group/model-exclusive policies always resolve to one.
    pub fn effective_max_in_flight_per_model(self, engine_limit: PhysicalInFlightLimit) -> usize {
        match self {
            Self::ExecutionGroupExclusive | Self::ModelExclusive => 1,
            Self::Concurrent {
                max_in_flight_per_model,
            } => max_in_flight_per_model.get().min(engine_limit.get()),
        }
    }
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
    /// Certification for overlap between separate physical calls. Missing or
    /// unsealed contracts remain execution-group exclusive.
    #[serde(default)]
    pub physical_launch_policy: PhysicalLaunchPolicy,
    pub recompute_safe: bool,
    /// The executor can synchronously prove that all model-owned cache state
    /// for an exact session has been released before recomputation or reuse.
    pub cache_release_safe: bool,
    pub prefix_reuse_safe: bool,
    pub max_batch_size: usize,
    /// Model-preferred number of decode inputs in one isolated scheduler
    /// transaction. The scheduler may reduce this for fairness or latency.
    #[serde(default = "default_preferred_decode_tokens")]
    pub preferred_decode_tokens: usize,
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
            physical_launch_policy: PhysicalLaunchPolicy::ExecutionGroupExclusive,
            recompute_safe: false,
            cache_release_safe: false,
            prefix_reuse_safe: false,
            max_batch_size: 1,
            preferred_decode_tokens: 1,
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

    /// Return the launch policy that may be consumed by physical admission.
    /// Catalog or fallback profiles cannot promote execution concurrency even
    /// if an unsealed policy value was copied into the profile.
    pub fn effective_physical_launch_policy(&self) -> PhysicalLaunchPolicy {
        if self.resolved_from_loaded_model {
            self.physical_launch_policy
        } else {
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        }
    }
}

const fn default_preferred_decode_tokens() -> usize {
    1
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
    pub adapter: Option<AdapterBindingKey>,
}

/// Canonical compatibility identity for one physical tensor-batch lane. Every
/// field participates in equality: models loaded on opposite sides of a reload
/// boundary, adapter upgrades, or incompatible tensor/state layouts can never
/// share one native batch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchLaneKey {
    pub execution_group: ExecutionGroupId,
    pub model_instance: ModelInstanceId,
    pub adapter_instance: AdapterInstanceId,
    pub adapter_abi: AdapterAbiRevision,
    pub capability_id: String,
    pub stage_id: StageId,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub compute_dtype: String,
    pub state_dtype: String,
    pub tensor_layout: String,
    pub quantization: String,
    pub state_schema: String,
    pub kernel_mode: String,
    pub semantic_mode: String,
    pub shape_bucket: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyQuantum {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub lane: BatchLaneKey,
    pub work: WorkUnit,
    pub cost: WorkCost,
    /// Optional shadow/managed-cache transaction fenced to this exact row.
    pub managed_cache: Option<ManagedCacheReservation>,
}

/// Backend-neutral identity of one row-level managed-cache reservation.
///
/// Physical tables and pins remain owned by the cache coordinator. Carrying
/// this compact fence in the execution envelope lets report validation reject
/// a receipt for another plan, request incarnation, arena generation, domain,
/// or table version before engine state is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManagedSessionGeneration(u64);

impl ManagedSessionGeneration {
    pub const INITIAL: Self = Self(1);

    pub const fn get(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn for_test(value: u64) -> Self {
        Self(value)
    }

    pub(crate) fn next(self) -> Result<Self> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or_else(|| Error::InferenceError("managed session generation overflow".into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedCacheReservation {
    pub txn_id: PlanId,
    pub session: SessionKey,
    pub session_generation: ManagedSessionGeneration,
    pub domains: Vec<ManagedCacheDomainReservation>,
    pub clocked_state: Option<ManagedClockedStateReservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedClockedStateReservation {
    model_instance: ModelInstanceId,
    pub sequence: u64,
    /// `None` is a legacy all-domain decoder-coupled transaction; `Some`
    /// authenticates the exact independently clocked selected groups.
    selections: Option<Arc<[TensorStateSelection]>>,
}

/// Compatibility name retained while decoder-coupled model adapters migrate
/// to the explicit clocked-state vocabulary.
pub type ManagedTensorStateReservation = ManagedClockedStateReservation;

impl ManagedClockedStateReservation {
    pub(crate) fn legacy(model_instance: ModelInstanceId, sequence: u64) -> Result<Self> {
        Self::new(model_instance, sequence, None)
    }

    pub(crate) fn selected(
        model_instance: ModelInstanceId,
        sequence: u64,
        selections: Arc<[TensorStateSelection]>,
    ) -> Result<Self> {
        if selections.is_empty() {
            return Err(Error::InvalidInput(
                "managed clocked-state reservation has no selected groups".into(),
            ));
        }
        Self::new(model_instance, sequence, Some(selections))
    }

    fn new(
        model_instance: ModelInstanceId,
        sequence: u64,
        selections: Option<Arc<[TensorStateSelection]>>,
    ) -> Result<Self> {
        if sequence == 0 {
            return Err(Error::InvalidInput(
                "managed clocked-state reservation has a zero sequence id".into(),
            ));
        }
        Ok(Self {
            model_instance,
            sequence,
            selections,
        })
    }

    pub(crate) const fn model_instance(&self) -> ModelInstanceId {
        self.model_instance
    }

    pub(crate) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(crate) fn selections(&self) -> Option<&[TensorStateSelection]> {
        self.selections.as_deref()
    }
}

/// One physical cache-domain transaction within a row reservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedCacheDomainReservation {
    pub arena: KvArenaId,
    pub domain: CacheDomainId,
    pub expected_version: u64,
    pub expected_committed_tokens: u32,
    /// Logical context already present in the provisional table when model
    /// execution starts. This may exceed the committed request-table length
    /// when admission attached immutable pages from the prefix index.
    pub execution_start_tokens: u32,
    pub target_committed_tokens: u32,
    pub target_window_start: u32,
    /// Hidden-token offset in the first provisional physical page.
    pub first_page_offset: u32,
    pub provisional_groups: Vec<GroupBlockTable>,
    pub writable_blocks: Vec<CacheBlockRef>,
}

impl ManagedCacheReservation {
    pub fn validate_for_row(&self, row: &ReadyQuantum) -> Result<()> {
        if self.txn_id != row.plan_id || self.session != row.session {
            return Err(Error::InvalidInput(
                "managed-cache reservation does not match its physical row".to_string(),
            ));
        }
        if self.domains.is_empty() && self.clocked_state.is_none() {
            return Err(Error::InvalidInput(
                "managed-cache reservation has no paged or clocked state".to_string(),
            ));
        }
        if let Some(clocked) = self.clocked_state.as_ref() {
            if clocked.model_instance() != row.lane.model_instance
                || self
                    .domains
                    .iter()
                    .any(|domain| domain.arena.model_instance != clocked.model_instance())
            {
                return Err(Error::InvalidInput(
                    "managed clocked-state reservation crossed its exact model instance".into(),
                ));
            }
        }
        let auxiliary = match &row.work {
            WorkUnit::SequenceStep {
                auxiliary_state, ..
            } => auxiliary_state.as_deref(),
            _ => None,
        };
        match (auxiliary, self.clocked_state.as_ref()) {
            (None, Some(reservation)) if reservation.selections().is_some() => {
                return Err(Error::InvalidInput(
                    "selected clocked-state reservation was attached to legacy sequence work"
                        .into(),
                ));
            }
            (Some(spans), None) if !spans.is_empty() => {
                return Err(Error::InvalidInput(
                    "clocked-state sequence spans have no physical reservation".into(),
                ));
            }
            (Some(spans), None) if spans.is_empty() => {}
            (Some(spans), Some(_)) if spans.is_empty() => {
                return Err(Error::InvalidInput(
                    "explicit empty clocked-state work cannot reserve a tensor transaction".into(),
                ));
            }
            (Some(spans), Some(reservation)) => {
                let selections = reservation.selections().ok_or_else(|| {
                    Error::InvalidInput(
                        "explicit clocked-state work received a legacy physical reservation".into(),
                    )
                })?;
                if spans.len() != selections.len() {
                    return Err(Error::InvalidInput(
                        "clocked-state reservation does not match the row's exact ordered spans"
                            .into(),
                    ));
                }
                for (span, selection) in spans.iter().zip(selections) {
                    let expected_cursor = u64::try_from(span.input().start).map_err(|_| {
                        Error::InvalidInput(
                            "clocked-state span start exceeds physical cursor width".into(),
                        )
                    })?;
                    let target_cursor = u64::try_from(span.input().end).map_err(|_| {
                        Error::InvalidInput(
                            "clocked-state span end exceeds physical cursor width".into(),
                        )
                    })?;
                    if selection.group != span.group()
                        || &selection.clock != span.clock()
                        || selection.expected_cursor != expected_cursor
                        || selection.target_cursor != target_cursor
                    {
                        return Err(Error::InvalidInput(
                            "clocked-state reservation does not match the row's exact ordered spans"
                                .into(),
                        ));
                    }
                }
            }
            _ => {}
        }
        if self
            .clocked_state
            .as_ref()
            .is_some_and(|reservation| reservation.sequence() == 0)
        {
            return Err(Error::InvalidInput(
                "managed tensor-state reservation has a zero sequence id".into(),
            ));
        }
        let mut identities = HashSet::with_capacity(self.domains.len());
        for domain in &self.domains {
            if domain.execution_start_tokens < domain.expected_committed_tokens
                || domain.target_committed_tokens < domain.execution_start_tokens
                || domain.target_window_start > domain.target_committed_tokens
                || domain.first_page_offset > domain.target_committed_tokens
            {
                return Err(Error::InvalidInput(
                    "managed-cache reservation has an invalid token range".to_string(),
                ));
            }
            if !identities.insert((domain.arena, domain.domain)) {
                return Err(Error::InvalidInput(
                    "managed-cache reservation repeats a cache domain".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Reconcile backend-sealed physical writes with this exact row.
    pub(crate) fn completed_write_receipt(
        &self,
        completions: &[Arc<KvWriteBatchCompletion>],
    ) -> Result<ManagedCacheReceipt> {
        self.completed_write_receipt_inner(completions, None)
    }

    /// Reconcile sealed backend writes with one common accepted prefix of the
    /// scheduler's maximum reservation.
    ///
    /// The receipt retains this exact reservation as its transaction fence.
    /// `committed_tokens` is an absolute logical cursor and must advance every
    /// domain beyond its execution start without exceeding any reserved target.
    pub(crate) fn completed_write_receipt_for_prefix(
        &self,
        completions: &[Arc<KvWriteBatchCompletion>],
        committed_tokens: u32,
    ) -> Result<ManagedCacheReceipt> {
        self.completed_write_receipt_inner(completions, Some(committed_tokens))
    }

    fn completed_write_receipt_inner(
        &self,
        completions: &[Arc<KvWriteBatchCompletion>],
        accepted_prefix: Option<u32>,
    ) -> Result<ManagedCacheReceipt> {
        if completions.is_empty() && !self.domains.is_empty() {
            return Err(Error::InferenceError(
                "managed-cache row returned no backend write completion".into(),
            ));
        }
        let mut receipts = Vec::with_capacity(self.domains.len());
        for domain in &self.domains {
            let committed_tokens = accepted_prefix.unwrap_or(domain.target_committed_tokens);
            if committed_tokens <= domain.execution_start_tokens
                || committed_tokens > domain.target_committed_tokens
            {
                return Err(Error::InferenceError(
                    "managed-cache accepted prefix is outside the reserved append range".into(),
                ));
            }
            let writable = domain
                .writable_blocks
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            if writable.len() != domain.writable_blocks.len() || writable.is_empty() {
                return Err(Error::InferenceError(
                    "managed-cache reservation has an invalid writable block set".into(),
                ));
            }
            let group = domain
                .provisional_groups
                .iter()
                .find(|table| writable.iter().any(|block| block.group == table.group))
                .ok_or_else(|| {
                    Error::InferenceError(
                        "managed-cache reservation has no table for its writable blocks".into(),
                    )
                })?;
            if writable.iter().any(|block| {
                block.arena != domain.arena
                    || block.group != group.group
                    || !group.blocks.contains(block)
            }) {
                return Err(Error::InferenceError(
                    "managed-cache writable blocks cross an arena or group fence".into(),
                ));
            }

            let matching = completions
                .iter()
                .filter(|completion| completion.arena() == domain.arena)
                .collect::<Vec<_>>();
            let page_tokens = matching
                .first()
                .map(|completion| completion.page_tokens())
                .ok_or_else(|| {
                    Error::InferenceError(
                        "managed-cache domain has no matching backend completion".into(),
                    )
                })?;
            if page_tokens == 0
                || matching
                    .iter()
                    .any(|completion| completion.page_tokens() != page_tokens)
            {
                return Err(Error::InferenceError(
                    "managed-cache completions disagree on page geometry".into(),
                ));
            }
            let expected = expected_domain_slots(domain, group, page_tokens, committed_tokens)?;
            let mut observed = HashSet::with_capacity(expected.len());
            for completion in matching {
                for slot in completion.slots() {
                    if writable.contains(&slot.block) && !observed.insert(*slot) {
                        return Err(Error::InferenceError(
                            "managed-cache physical slot was acknowledged more than once".into(),
                        ));
                    }
                }
            }
            if observed != expected {
                return Err(Error::InferenceError(
                    "managed-cache completion does not match the row's exact physical slots".into(),
                ));
            }
            let written_blocks = domain
                .writable_blocks
                .iter()
                .copied()
                .filter(|block| expected.iter().any(|slot| slot.block == *block))
                .collect::<Vec<_>>();
            if written_blocks.is_empty() {
                return Err(Error::InferenceError(
                    "managed-cache accepted prefix has no writable physical page".into(),
                ));
            }
            receipts.push(ManagedCacheDomainReceipt {
                arena: domain.arena,
                domain: domain.domain,
                written_blocks,
                page_tokens,
            });
        }
        Ok(ManagedCacheReceipt {
            reservation: self.clone(),
            domains: receipts,
            accepted_prefix,
            clocked_state: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn completed_write_receipt_for_test(&self) -> ManagedCacheReceipt {
        ManagedCacheReceipt {
            reservation: self.clone(),
            domains: self
                .domains
                .iter()
                .map(|domain| ManagedCacheDomainReceipt {
                    arena: domain.arena,
                    domain: domain.domain,
                    written_blocks: domain.writable_blocks.clone(),
                    page_tokens: inferred_page_tokens_for_test(domain),
                })
                .collect(),
            accepted_prefix: None,
            clocked_state: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn completed_write_receipt_for_prefix_for_test(
        &self,
        committed_tokens: u32,
        page_tokens: u32,
    ) -> Result<ManagedCacheReceipt> {
        let mut domains = Vec::with_capacity(self.domains.len());
        for domain in &self.domains {
            if committed_tokens <= domain.execution_start_tokens
                || committed_tokens > domain.target_committed_tokens
            {
                return Err(Error::InvalidInput(
                    "test managed-cache prefix is outside the reservation".into(),
                ));
            }
            let writable = domain
                .writable_blocks
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let group = domain
                .provisional_groups
                .iter()
                .find(|group| writable.iter().any(|block| block.group == group.group))
                .ok_or_else(|| {
                    Error::InvalidInput("test reservation has no writable group".into())
                })?;
            let expected = expected_domain_slots(domain, group, page_tokens, committed_tokens)?;
            let written_blocks = domain
                .writable_blocks
                .iter()
                .copied()
                .filter(|block| expected.iter().any(|slot| slot.block == *block))
                .collect::<Vec<_>>();
            domains.push(ManagedCacheDomainReceipt {
                arena: domain.arena,
                domain: domain.domain,
                written_blocks,
                page_tokens,
            });
        }
        Ok(ManagedCacheReceipt {
            reservation: self.clone(),
            domains,
            accepted_prefix: Some(committed_tokens),
            clocked_state: None,
        })
    }
}

fn expected_domain_slots(
    domain: &ManagedCacheDomainReservation,
    table: &GroupBlockTable,
    page_tokens: u32,
    committed_tokens: u32,
) -> Result<HashSet<KvSlotRef>> {
    if committed_tokens <= domain.execution_start_tokens
        || committed_tokens > domain.target_committed_tokens
    {
        return Err(Error::InferenceError(
            "managed-cache reservation has no physical append range".into(),
        ));
    }
    let first_logical_page = domain.target_window_start / page_tokens;
    let mut slots =
        HashSet::with_capacity((committed_tokens - domain.execution_start_tokens) as usize);
    for position in domain.execution_start_tokens..committed_tokens {
        let logical_page = position / page_tokens;
        let table_index = logical_page
            .checked_sub(first_logical_page)
            .ok_or_else(|| {
                Error::InferenceError("managed-cache append precedes its physical window".into())
            })?;
        let block = table
            .blocks
            .get(table_index as usize)
            .copied()
            .ok_or_else(|| {
                Error::InferenceError("managed-cache append exceeds its physical table".into())
            })?;
        slots.insert(KvSlotRef {
            block,
            offset: position % page_tokens,
        });
    }
    Ok(slots)
}

/// Physical completion acknowledgement for one managed-cache row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedCacheReceipt {
    pub reservation: ManagedCacheReservation,
    pub domains: Vec<ManagedCacheDomainReceipt>,
    // `None` preserves the legacy exact-full-reservation receipt. `Some` is a
    // common accepted cursor authenticated from sealed backend completions.
    accepted_prefix: Option<u32>,
    clocked_state: Option<ManagedClockedStateReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedClockedStateReceipt {
    completion: TensorStateBatchCompletion,
}

impl ManagedClockedStateReceipt {
    pub(crate) fn new(completion: TensorStateBatchCompletion) -> Self {
        Self { completion }
    }

    pub(crate) const fn completion(&self) -> &TensorStateBatchCompletion {
        &self.completion
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedCacheDomainReceipt {
    pub arena: KvArenaId,
    pub domain: CacheDomainId,
    pub written_blocks: Vec<CacheBlockRef>,
    page_tokens: u32,
}

impl ManagedCacheReceipt {
    pub(crate) const fn accepted_prefix(&self) -> Option<u32> {
        self.accepted_prefix
    }

    pub(crate) const fn clocked_state(&self) -> Option<&ManagedClockedStateReceipt> {
        self.clocked_state.as_ref()
    }

    pub(crate) fn with_clocked_state_completion(
        mut self,
        completion: TensorStateBatchCompletion,
    ) -> Result<Self> {
        if self.clocked_state.is_some() {
            return Err(Error::InvalidInput(
                "managed receipt already carries a clocked-state completion".into(),
            ));
        }
        let reservation = self.reservation.clocked_state.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                "clocked-state completion has no matching managed reservation".into(),
            )
        })?;
        let selected = reservation.selections().ok_or_else(|| {
            Error::InvalidInput(
                "legacy clocked-state reservation cannot accept a selected completion".into(),
            )
        })?;
        if completion.sequence().get() != reservation.sequence()
            || completion.selections().ne(selected.iter())
        {
            return Err(Error::InvalidInput(
                "clocked-state completion does not match its exact managed reservation".into(),
            ));
        }
        self.clocked_state = Some(ManagedClockedStateReceipt::new(completion));
        Ok(self)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match (
            self.reservation.clocked_state.as_ref(),
            self.clocked_state.as_ref(),
        ) {
            (Some(reservation), Some(receipt)) => {
                let selected = reservation.selections().ok_or_else(|| {
                    Error::InferenceError(
                        "legacy clocked-state reservation cannot carry a selected completion"
                            .into(),
                    )
                })?;
                if receipt.completion().sequence().get() != reservation.sequence()
                    || receipt.completion().selections().ne(selected.iter())
                {
                    return Err(Error::InferenceError(
                        "clocked-state receipt does not match its exact reservation".into(),
                    ));
                }
            }
            (Some(reservation), None) if reservation.selections().is_some() => {
                return Err(Error::InferenceError(
                    "selected clocked-state reservation has no arena completion".into(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::InferenceError(
                    "clocked-state receipt has no matching reservation".into(),
                ));
            }
            _ => {}
        }
        if self.domains.len() != self.reservation.domains.len() {
            return Err(Error::InferenceError(
                "managed-cache receipt does not cover every reserved domain".to_string(),
            ));
        }
        let mut identities = HashSet::with_capacity(self.domains.len());
        for receipt in &self.domains {
            let reservation = self
                .reservation
                .domains
                .iter()
                .find(|reserved| {
                    reserved.arena == receipt.arena && reserved.domain == receipt.domain
                })
                .ok_or_else(|| {
                    Error::InferenceError(
                        "managed-cache receipt contains a foreign cache domain".to_string(),
                    )
                })?;
            if !identities.insert((receipt.arena, receipt.domain)) {
                return Err(Error::InferenceError(
                    "managed-cache receipt repeats a cache domain".to_string(),
                ));
            }
            let blocks = receipt
                .written_blocks
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let committed_tokens = self
                .accepted_prefix
                .unwrap_or(reservation.target_committed_tokens);
            if committed_tokens <= reservation.execution_start_tokens
                || committed_tokens > reservation.target_committed_tokens
                || receipt.page_tokens == 0
            {
                return Err(Error::InferenceError(
                    "managed-cache receipt has an invalid committed prefix".into(),
                ));
            }
            let writable = reservation
                .writable_blocks
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let group = reservation
                .provisional_groups
                .iter()
                .find(|group| writable.iter().any(|block| block.group == group.group))
                .ok_or_else(|| {
                    Error::InferenceError(
                        "managed-cache receipt has no reserved writable group".into(),
                    )
                })?;
            let expected_blocks =
                expected_domain_slots(reservation, group, receipt.page_tokens, committed_tokens)?
                    .into_iter()
                    .map(|slot| slot.block)
                    .filter(|block| writable.contains(block))
                    .collect::<HashSet<_>>();
            if blocks.len() != receipt.written_blocks.len()
                || blocks != expected_blocks
                || receipt
                    .written_blocks
                    .iter()
                    .any(|block| block.arena != receipt.arena)
            {
                return Err(Error::InferenceError(
                    "managed-cache receipt does not match the domain writable set".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
fn inferred_page_tokens_for_test(domain: &ManagedCacheDomainReservation) -> u32 {
    let upper = domain.target_committed_tokens.max(1);
    (1..=upper)
        .find(|page_tokens| {
            domain.target_window_start % page_tokens == domain.first_page_offset
                && domain.provisional_groups.iter().any(|group| {
                    expected_domain_slots(
                        domain,
                        group,
                        *page_tokens,
                        domain.target_committed_tokens,
                    )
                    .is_ok_and(|slots| {
                        let blocks = slots
                            .into_iter()
                            .map(|slot| slot.block)
                            .collect::<HashSet<_>>();
                        domain
                            .writable_blocks
                            .iter()
                            .all(|block| blocks.contains(block))
                    })
                })
        })
        .unwrap_or(upper)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalBatch {
    pub batch_id: BatchId,
    pub lane: BatchLaneKey,
    pub mode: NativeBatchMode,
    pub budget: BatchBudget,
    pub rows: Vec<ReadyQuantum>,
    /// Materialized elements including padding. Ragged/packed adapters report
    /// the useful tensor element count here.
    pub materialized_tensor_elements: u64,
    pub workspace: ResourceVector,
}

impl PhysicalBatch {
    pub fn expected_dispatch(&self) -> BatchDispatch {
        let width = self.rows.len().max(1);
        match self.mode {
            NativeBatchMode::Static => BatchDispatch::new(BatchDispatchKind::TensorStatic, width),
            NativeBatchMode::Continuous => {
                BatchDispatch::new(BatchDispatchKind::TensorContinuous, width)
            }
            NativeBatchMode::None if width > 1 => {
                BatchDispatch::new(BatchDispatchKind::RequestParallel, width)
            }
            NativeBatchMode::None => BatchDispatch::serial(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.budget.validate()?;
        if self.rows.is_empty() {
            return Err(Error::InvalidInput(
                "physical batch cannot be empty".to_string(),
            ));
        }
        if self.mode == NativeBatchMode::None
            && self.rows.len() > 1
            && self.budget.max_rows < self.rows.len()
        {
            return Err(Error::InvalidInput(
                "request-parallel physical dispatch exceeds its declared width".to_string(),
            ));
        }

        let mut keys = HashSet::with_capacity(self.rows.len());
        let mut cost = WorkCost::default();
        for (row_count, row) in self.rows.iter().enumerate() {
            if row.lane != self.lane {
                return Err(Error::InvalidInput(
                    "physical batch contains an incompatible lane".to_string(),
                ));
            }
            if !keys.insert((row.session.clone(), row.plan_id)) {
                return Err(Error::InvalidInput(
                    "physical batch contains a duplicate session plan".to_string(),
                ));
            }
            if !self.budget.admits(row_count, cost, row.cost) {
                return Err(Error::InvalidInput(
                    "physical batch exceeds its declared work budget".to_string(),
                ));
            }
            cost = cost.checked_add(row.cost).ok_or_else(|| {
                Error::InvalidInput("physical batch work accounting overflowed".to_string())
            })?;
        }

        if self.materialized_tensor_elements < cost.tensor_elements {
            return Err(Error::InvalidInput(
                "physical batch materialization is smaller than useful tensor work".to_string(),
            ));
        }
        let workspace_bytes = self.workspace.workspace_bytes()?;
        if workspace_bytes < cost.workspace.workspace_bytes()? {
            return Err(Error::InvalidInput(
                "physical batch workspace is smaller than its row estimates".to_string(),
            ));
        }
        if workspace_bytes > self.budget.max_workspace_bytes {
            return Err(Error::InvalidInput(
                "physical batch workspace exceeds its declared budget".to_string(),
            ));
        }
        let padded = self
            .materialized_tensor_elements
            .saturating_sub(cost.tensor_elements);
        if cost.tensor_elements == 0 {
            if padded > 0 {
                return Err(Error::InvalidInput(
                    "physical batch cannot pad empty tensor work".to_string(),
                ));
            }
        } else if u128::from(padded) * 10_000
            > u128::from(cost.tensor_elements) * u128::from(self.budget.max_padding_basis_points)
        {
            return Err(Error::InvalidInput(
                "physical batch exceeds its declared padding budget".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateDisposition {
    Unchanged,
    ValidNext,
    RolledBack,
    RestartPending,
    Poisoned,
}

#[derive(Debug, Clone)]
pub struct PhysicalBatchRowReport {
    pub execution: ExecutionReport,
    pub state: StateDisposition,
    pub managed_cache: Option<ManagedCacheReceipt>,
}

impl PhysicalBatchRowReport {
    fn validate_state(&self) -> Result<()> {
        match &self.execution.disposition {
            ExecutionDisposition::Progress | ExecutionDisposition::Yielded(_)
                if self.state != StateDisposition::ValidNext =>
            {
                return Err(Error::InferenceError(
                    "continuing execution must publish valid next model state".to_string(),
                ));
            }
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::RetrySameSession
                    && !matches!(
                        self.state,
                        StateDisposition::Unchanged | StateDisposition::RolledBack
                    ) =>
            {
                return Err(Error::InferenceError(
                    "same-session retry requires unchanged or rolled-back model state".to_string(),
                ));
            }
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::Recompute
                    && self.state == StateDisposition::ValidNext =>
            {
                return Err(Error::InferenceError(
                    "recompute retry cannot publish advanced model state".to_string(),
                ));
            }
            ExecutionDisposition::RestartSequence(_)
                if self.state != StateDisposition::RestartPending =>
            {
                return Err(Error::InferenceError(
                    "sequence restart requires rolled-back restart-pending model state".to_string(),
                ));
            }
            _ if self.state == StateDisposition::RestartPending
                && !matches!(
                    self.execution.disposition,
                    ExecutionDisposition::RestartSequence(_)
                ) =>
            {
                return Err(Error::InferenceError(
                    "restart-pending model state requires a sequence restart outcome".to_string(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalBatchReport {
    pub batch_id: BatchId,
    pub lane: BatchLaneKey,
    pub dispatch: BatchDispatch,
    pub observed_resources: ResourceVector,
    pub elapsed: Duration,
    pub rows: Vec<PhysicalBatchRowReport>,
}

impl PhysicalBatchReport {
    pub fn validate_against(
        &self,
        batch: &PhysicalBatch,
        active_plans: &HashMap<PlanId, ExecutionPlan>,
    ) -> Result<()> {
        batch.validate()?;
        if self.batch_id != batch.batch_id || self.lane != batch.lane {
            return Err(Error::InferenceError(
                "physical batch report does not match its dispatch envelope".to_string(),
            ));
        }
        if self.dispatch.width != batch.rows.len() || self.rows.len() != batch.rows.len() {
            return Err(Error::InferenceError(
                "physical batch report width does not match its planned rows".to_string(),
            ));
        }
        self.observed_resources.workspace_bytes()?;
        if !self.observed_resources.fits_within(batch.workspace) {
            return Err(Error::InferenceError(
                "physical batch used more workspace than it reserved".to_string(),
            ));
        }
        match self.dispatch.kind {
            BatchDispatchKind::NotDispatched
                if self.rows.iter().any(|row| {
                    !matches!(
                        row.execution.disposition,
                        ExecutionDisposition::Failed(_)
                            | ExecutionDisposition::Finished(
                                FinishReason::Cancelled
                                    | FinishReason::TimedOut
                                    | FinishReason::Rejected
                            )
                    )
                }) =>
            {
                return Err(Error::InferenceError(
                    "a non-dispatched batch may only fail or terminalize rows before model entry"
                        .to_string(),
                ));
            }
            BatchDispatchKind::Serial if batch.rows.len() != 1 => {
                return Err(Error::InferenceError(
                    "serial physical dispatch must have width one".to_string(),
                ));
            }
            BatchDispatchKind::TensorStatic if batch.mode != NativeBatchMode::Static => {
                return Err(Error::InferenceError(
                    "physical batch reported undeclared static tensor execution".to_string(),
                ));
            }
            BatchDispatchKind::TensorContinuous if batch.mode != NativeBatchMode::Continuous => {
                return Err(Error::InferenceError(
                    "physical batch reported undeclared continuous tensor execution".to_string(),
                ));
            }
            BatchDispatchKind::RequestParallel
                if batch.mode != NativeBatchMode::None || batch.rows.len() < 2 =>
            {
                return Err(Error::InferenceError(
                    "request-parallel dispatch requires a multi-row non-tensor batch".to_string(),
                ));
            }
            _ => {}
        }

        let expected = batch
            .rows
            .iter()
            .map(|row| ((row.session.clone(), row.plan_id), row))
            .collect::<HashMap<_, _>>();
        let mut reported = HashSet::with_capacity(self.rows.len());
        for row in &self.rows {
            let key = (row.execution.session.clone(), row.execution.plan_id);
            if !reported.insert(key.clone()) {
                return Err(Error::InferenceError(
                    "physical batch report contains a duplicate session plan".to_string(),
                ));
            }
            if !expected.contains_key(&key) {
                return Err(Error::InferenceError(
                    "physical batch report contains a foreign session plan".to_string(),
                ));
            }
            if row.execution.dispatch != self.dispatch {
                return Err(Error::InferenceError(
                    "physical batch row disagrees with envelope dispatch metadata".to_string(),
                ));
            }
            let plan = active_plans.get(&row.execution.plan_id).ok_or_else(|| {
                Error::InferenceError(
                    "physical batch report references an inactive execution plan".to_string(),
                )
            })?;
            row.execution.validate_against(plan)?;
            row.validate_state()?;
            let expected_row = expected
                .get(&key)
                .expect("reported row was validated above");
            if matches!(
                row.execution.disposition,
                ExecutionDisposition::RestartSequence(_)
            ) {
                if expected_row.managed_cache.is_none() {
                    return Err(Error::InferenceError(
                        "sequence restart requires an exact managed-cache reservation".to_string(),
                    ));
                }
                if row.managed_cache.is_some() {
                    return Err(Error::InferenceError(
                        "sequence restart cannot publish a managed-cache receipt".to_string(),
                    ));
                }
            }
            match (&expected_row.managed_cache, &row.managed_cache) {
                (None, None) => {}
                (None, Some(_)) => {
                    return Err(Error::InferenceError(
                        "physical batch report contains an unplanned managed-cache receipt"
                            .to_string(),
                    ));
                }
                (Some(reservation), Some(receipt)) => {
                    if &receipt.reservation != reservation {
                        return Err(Error::InferenceError(
                            "managed-cache receipt does not match its exact row reservation"
                                .to_string(),
                        ));
                    }
                    receipt.validate()?;
                }
                (Some(_), None) if row.state == StateDisposition::ValidNext => {
                    return Err(Error::InferenceError(
                        "continuing managed-cache row omitted its physical write receipt"
                            .to_string(),
                    ));
                }
                (Some(_), None) => {
                    // Failed, rolled-back, poisoned, or terminal rows abort the
                    // reservation instead of publishing cache state.
                }
            }
        }
        if reported.len() != expected.len() {
            return Err(Error::InferenceError(
                "physical batch report omitted a planned session".to_string(),
            ));
        }
        Ok(())
    }
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
    Cancelled,
    TimedOut,
    Rejected,
}

/// Model-authored reason for discarding one committed generation attempt and
/// rebuilding the same logical sequence from context zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceRestartReason {
    ModelFallback,
}

impl FinishReason {
    fn terminal_outcome(self) -> TerminalOutcome {
        match self {
            Self::Completed => TerminalOutcome::Completed,
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
    Row,
    PhysicalBatch,
    ExecutionGroup,
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
            scope: FailureScope::Row,
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
    RestartSequence(SequenceRestartReason),
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
    RestartPending,
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
                    Prefilling | Decoding | Cancelling | PreemptedRecompute | RestartPending
                )
                | (
                    Decoding,
                    Decoding | Cancelling | PreemptedRecompute | RestartPending
                )
                | (AtomicRunning, Cancelling)
                | (PipelineRunning, PipelineRunning | Cancelling)
                | (PreemptedRecompute, Admitted | Prefilling | Cancelling)
                | (RestartPending, Cancelling)
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
    pub stage: Option<StageDescriptor>,
}

#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub input_consumed: usize,
    pub output_produced: usize,
    pub observed_resources: ResourceVector,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
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
        validate_plan_clocked_state(plan)?;
        match self.dispatch.kind {
            BatchDispatchKind::NotDispatched
                if !matches!(
                    self.disposition,
                    ExecutionDisposition::Failed(_)
                        | ExecutionDisposition::Finished(
                            FinishReason::Cancelled
                                | FinishReason::TimedOut
                                | FinishReason::Rejected
                        )
                ) =>
            {
                return Err(Error::InferenceError(
                    "non-dispatched execution must fail or terminalize before model entry"
                        .to_string(),
                ));
            }
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
        if self.dispatch.kind == BatchDispatchKind::NotDispatched
            && self.provenance.dispatch_state != DispatchState::NotStarted
        {
            return Err(Error::InferenceError(
                "non-dispatched execution cannot claim model entry".to_string(),
            ));
        }
        if self.dispatch.kind != BatchDispatchKind::NotDispatched
            && self.provenance.dispatch_state == DispatchState::NotStarted
            && !(self.dispatch.kind == BatchDispatchKind::RequestParallel
                && matches!(
                    self.disposition,
                    ExecutionDisposition::Finished(
                        FinishReason::Cancelled | FinishReason::TimedOut | FinishReason::Rejected
                    )
                ))
        {
            return Err(Error::InferenceError(
                "dispatched execution must record model entry unless an independent row terminated before entry"
                    .to_string(),
            ));
        }
        if self.provenance.deadline_phase.is_some()
            != matches!(
                self.disposition,
                ExecutionDisposition::Finished(FinishReason::TimedOut)
            )
        {
            return Err(Error::InferenceError(
                "deadline provenance must match a timed-out disposition".to_string(),
            ));
        }
        if self.provenance.failure_origin.is_some()
            != matches!(self.disposition, ExecutionDisposition::Failed(_))
        {
            return Err(Error::InferenceError(
                "failure provenance must match a failed disposition".to_string(),
            ));
        }
        match plan.work {
            WorkUnit::PreSequencePreparation { .. } => {
                return Err(Error::InferenceError(
                    "pre-sequence preparation must complete before EngineCore admission"
                        .to_string(),
                ));
            }
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
            ExecutionDisposition::Progress
            | ExecutionDisposition::Yielded(_)
            | ExecutionDisposition::RestartSequence(_) => {
                if self.output_finished || self.output_has_error {
                    return Err(Error::InferenceError(
                        "non-terminal execution returned a terminal or errored payload".to_string(),
                    ));
                }
                if matches!(self.disposition, ExecutionDisposition::RestartSequence(_)) {
                    if !self.safe_point {
                        return Err(Error::InferenceError(
                            "sequence restart requires a declared safe point".to_string(),
                        ));
                    }
                    if self.input_consumed != 0 || self.output_produced != 0 {
                        return Err(Error::InferenceError(
                            "sequence restart cannot report committed progress".to_string(),
                        ));
                    }
                    if !matches!(plan.work, WorkUnit::SequenceStep { .. }) {
                        return Err(Error::InferenceError(
                            "sequence restart is only valid for sequence execution".to_string(),
                        ));
                    }
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

fn validate_plan_clocked_state(plan: &ExecutionPlan) -> Result<()> {
    let WorkUnit::SequenceStep {
        phase,
        auxiliary_state,
        ..
    } = &plan.work
    else {
        return Ok(());
    };
    let policy = plan
        .stage
        .as_ref()
        .and_then(|stage| stage.retained_state_selections.as_deref());
    match (policy, auxiliary_state.as_deref()) {
        (None, None) => return Ok(()),
        (Some(_), Some(spans)) if *phase == SequencePhase::Decode && !spans.is_empty() => {
            return Err(Error::InferenceError(
                "sequence decode cannot advance auxiliary retained-state clocks".into(),
            ));
        }
        (Some(selections), Some(spans)) => {
            let mut previous_group = None;
            for span in spans {
                if span.input().is_empty()
                    || previous_group.is_some_and(|previous| previous >= span.group().get())
                {
                    return Err(Error::InferenceError(
                        "sequence auxiliary retained-state spans are not canonical and unique"
                            .into(),
                    ));
                }
                if selections
                    .binary_search_by_key(&span.group().get(), |selection| selection.group().get())
                    .ok()
                    .and_then(|index| selections.get(index))
                    .is_none_or(|selection| selection.clock() != span.clock())
                {
                    return Err(Error::InferenceError(
                        "sequence auxiliary retained-state span is not authorized by its exact stage"
                            .into(),
                    ));
                }
                previous_group = Some(span.group().get());
            }
            return Ok(());
        }
        _ => {}
    }
    Err(Error::InferenceError(
        "sequence auxiliary retained-state policy was not sealed into its plan".into(),
    ))
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

    /// Release a plan that never entered model execution. Committed progress
    /// and the request's lifecycle state are unchanged; the next scheduler
    /// cycle may prepare a fresh plan identity for the same safe point.
    pub(crate) fn rollback_unexecuted_plan(&mut self, plan_id: PlanId) -> bool {
        if self.active_plan != Some(plan_id) {
            return false;
        }
        self.active_plan = None;
        true
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
            ExecutionDisposition::RestartSequence(_) => Some(ExecutionState::RestartPending),
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

    fn lane() -> BatchLaneKey {
        BatchLaneKey {
            execution_group: ExecutionGroupId::new(1),
            model_instance: ModelInstanceId::new(2),
            adapter_instance: AdapterInstanceId::new(3),
            adapter_abi: AdapterAbiRevision::new(1),
            capability_id: "chat".to_string(),
            stage_id: StageId::new(4),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: "dense".to_string(),
            quantization: "none".to_string(),
            state_schema: "test.v1".to_string(),
            kernel_mode: "reference".to_string(),
            semantic_mode: "greedy".to_string(),
            shape_bucket: "tokens.1".to_string(),
        }
    }

    #[test]
    fn execution_id_newtypes_do_not_alias_domains() {
        let group = ExecutionGroupId::new(7);
        let model = ModelInstanceId::new(7);
        let adapter = AdapterInstanceId::new(7);
        let stage = StageId::new(7);
        let batch = BatchId::new(7);

        assert_eq!(group.get(), 7);
        assert_eq!(model.get(), 7);
        assert_eq!(adapter.get(), 7);
        assert_eq!(stage.get(), 7);
        assert_eq!(batch.get(), 7);
        assert_eq!(AdapterAbiRevision::new(1).get(), 1);
    }

    #[test]
    fn clocked_state_projection_maps_split_quanta_exactly() {
        let projection = ClockedStateProjection::new(
            InputRange::new(10, 14).unwrap(),
            ClockedStateSelection::new(StateGroupId::new(2), StateClock::AudioSamples).unwrap(),
            InputRange::new(1_000, 1_640).unwrap(),
        )
        .unwrap();
        let first = projection
            .project(InputRange::new(8, 12).unwrap())
            .unwrap()
            .unwrap();
        let second = projection
            .project(InputRange::new(12, 16).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(first.input(), InputRange::new(1_000, 1_320).unwrap());
        assert_eq!(second.input(), InputRange::new(1_320, 1_640).unwrap());
        assert!(projection
            .project(InputRange::new(14, 15).unwrap())
            .unwrap()
            .is_none());
    }

    #[test]
    fn stage_clocked_state_authorization_is_canonical_and_tri_state() {
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "prefill",
            &profile,
            NativeBatchMode::None,
        );
        assert!(stage.retained_state_selections.is_none());
        stage.retained_state_selections = Some(vec![]);
        stage.validate().unwrap();
        stage.retained_state_selections = Some(vec![
            ClockedStateSelection::new(StateGroupId::new(2), StateClock::AudioSamples).unwrap(),
            ClockedStateSelection::new(StateGroupId::new(1), StateClock::EncoderTokens).unwrap(),
        ]);
        assert!(stage.validate().is_err());
    }

    #[test]
    fn legacy_stage_descriptor_stays_fail_closed_at_width_one() {
        let profile = ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Atomic);
        let stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "legacy",
            &profile,
            NativeBatchMode::None,
        );

        assert_eq!(stage.max_batch_size, 1);
        assert_eq!(stage.batch_mode, NativeBatchMode::None);
        assert_eq!(stage.progress, StageProgressKind::Atomic);
        assert_eq!(
            stage.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
        assert_eq!(
            profile.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
        assert!(stage.validate().is_ok());
    }

    #[test]
    fn tensor_batchability_does_not_certify_overlapping_physical_launches() {
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cuda, None, ExecutionMode::Atomic);
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.max_batch_size = 8;

        assert_eq!(
            profile.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
        assert_eq!(
            profile
                .physical_launch_policy
                .effective_max_in_flight_per_model(PhysicalInFlightLimit::new(8).unwrap()),
            1
        );

        profile.resolved_from_loaded_model = true;
        let stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "tensor.batchable",
            &profile,
            NativeBatchMode::Static,
        );
        assert_eq!(stage.concurrency, ConcurrencyClass::Batchable);
        assert_eq!(
            stage.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
        let mut falsely_concurrent = stage;
        falsely_concurrent.physical_launch_policy = PhysicalLaunchPolicy::concurrent(8).unwrap();
        assert!(falsely_concurrent.validate().is_err());
    }

    #[test]
    fn unresolved_profiles_cannot_promote_physical_launches() {
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Atomic);
        profile.physical_launch_policy = PhysicalLaunchPolicy::concurrent(4).unwrap();

        assert_eq!(
            profile.effective_physical_launch_policy(),
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );

        profile.resolved_from_loaded_model = true;
        assert_eq!(
            profile.effective_physical_launch_policy(),
            PhysicalLaunchPolicy::concurrent(4).unwrap()
        );
    }

    #[test]
    fn physical_launch_policy_clamps_model_and_engine_capacity_axes() {
        let engine_limit = PhysicalInFlightLimit::new(4).unwrap();
        assert_eq!(
            PhysicalLaunchPolicy::ExecutionGroupExclusive
                .effective_max_in_flight_per_model(engine_limit),
            1
        );
        assert_eq!(
            PhysicalLaunchPolicy::ModelExclusive.effective_max_in_flight_per_model(engine_limit),
            1
        );
        assert_eq!(
            PhysicalLaunchPolicy::concurrent(3)
                .unwrap()
                .effective_max_in_flight_per_model(engine_limit),
            3
        );
        assert_eq!(
            PhysicalLaunchPolicy::concurrent(8)
                .unwrap()
                .effective_max_in_flight_per_model(engine_limit),
            4
        );
        assert!(PhysicalLaunchPolicy::concurrent(0).is_err());
    }

    #[test]
    fn physical_launch_policy_deserialization_fails_closed() {
        let profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            Some(ModelVariant::Qwen306B),
            ExecutionMode::Atomic,
        );
        let mut legacy = serde_json::to_value(&profile).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("physical_launch_policy");
        let legacy: ExecutionProfile = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            legacy.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );

        let stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "legacy-stage",
            &profile,
            NativeBatchMode::None,
        );
        let mut legacy_stage = serde_json::to_value(&stage).unwrap();
        legacy_stage
            .as_object_mut()
            .unwrap()
            .remove("physical_launch_policy");
        let legacy_stage: StageDescriptor = serde_json::from_value(legacy_stage).unwrap();
        assert_eq!(
            legacy_stage.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );

        for invalid in [
            r#"{"kind":"concurrent","max_in_flight_per_model":0}"#,
            r#"{"kind":"backend_default"}"#,
        ] {
            assert!(serde_json::from_str::<PhysicalLaunchPolicy>(invalid).is_err());
        }
    }

    #[test]
    fn adapter_routes_work_to_exact_model_owned_stages() {
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "text.prefill",
            &profile,
            NativeBatchMode::Static,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        let mut decode = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "text.decode",
            &profile,
            NativeBatchMode::Continuous,
        );
        decode.selector = StageWorkSelector::SequenceDecode;
        decode.progress = StageProgressKind::Iterative;
        decode.membership_safe_point = MembershipSafePoint::QuantumBoundary;
        let binding = ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: ModelInstanceId::new(2),
            adapter_instance_id: AdapterInstanceId::new(3),
            adapter_abi_revision: AdapterAbiRevision::new(1),
            model_variant: ModelVariant::Qwen306B,
            capability_id: "chat".to_string(),
            stages: Arc::from([prefill, decode]),
        };
        binding.validate().unwrap();

        let prefill_work = WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange { start: 0, end: 8 },
            max_output_steps: 1,
            auxiliary_state: None,
        };
        let decode_work = WorkUnit::SequenceStep {
            phase: SequencePhase::Decode,
            input: InputRange { start: 8, end: 9 },
            max_output_steps: 1,
            auxiliary_state: None,
        };
        assert_eq!(
            binding.stage_for_work(&prefill_work).unwrap().id,
            StageId::new(1)
        );
        assert_eq!(
            binding.stage_for_work(&decode_work).unwrap().id,
            StageId::new(2)
        );
    }

    #[test]
    fn preparation_selector_is_distinct_from_decoder_prefill() {
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        let mut preparation = StageDescriptor::from_execution_profile(
            StageId::new(9),
            "audio.encoder.prepare",
            &profile,
            NativeBatchMode::Static,
        );
        preparation.selector = StageWorkSelector::PreSequencePreparation;
        preparation.progress = StageProgressKind::Atomic;
        preparation.shape_policy = StageShapePolicy::Ragged;
        preparation.max_padding_basis_points = 0;
        preparation.membership_safe_point = MembershipSafePoint::OperationBoundary;
        let mut prefill = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "decoder.prefill",
            &profile,
            NativeBatchMode::None,
        );
        prefill.selector = StageWorkSelector::SequencePrefill;
        let binding = ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: ModelInstanceId::new(2),
            adapter_instance_id: AdapterInstanceId::new(3),
            adapter_abi_revision: AdapterAbiRevision::new(1),
            model_variant: ModelVariant::Qwen306B,
            capability_id: "asr".to_string(),
            stages: Arc::from([preparation, prefill]),
        };
        binding.validate().unwrap();

        let preparation_work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.audio".to_string(),
        };
        let prefill_work = WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange { start: 0, end: 1 },
            max_output_steps: 0,
            auxiliary_state: None,
        };
        assert_eq!(
            binding.stage_for_work(&preparation_work).unwrap().id,
            StageId::new(9)
        );
        assert_eq!(
            binding.stage_for_work(&prefill_work).unwrap().id,
            StageId::new(1)
        );
    }

    #[test]
    fn continuous_stage_requires_repeatable_safe_points() {
        let invalid = StageDescriptor {
            id: StageId::new(2),
            name: "atomic".to_string(),
            selector: StageWorkSelector::Atomic,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
            concurrency: ConcurrencyClass::Batchable,
            physical_launch_policy: PhysicalLaunchPolicy::ExecutionGroupExclusive,
            batch_mode: NativeBatchMode::Continuous,
            max_batch_size: 2,
            max_work_units: 2,
            workspace_base_bytes: 0,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes: 1,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Ragged,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        };
        assert!(invalid.validate().is_err());

        let valid = StageDescriptor {
            progress: StageProgressKind::Iterative,
            membership_safe_point: MembershipSafePoint::QuantumBoundary,
            ..invalid
        };
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn incremental_output_visibility_is_restricted_to_non_tensor_stages() {
        let profile = ExecutionProfile::fail_closed(
            BackendKind::Cpu,
            Some(ModelVariant::Qwen306B),
            ExecutionMode::Atomic,
        );
        let mut compatibility = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "chat.compatibility",
            &profile,
            NativeBatchMode::None,
        );
        compatibility.output_visibility = OutputVisibility::IncrementalCommitted;
        assert!(compatibility.validate().is_ok());

        let mut tensor = StageDescriptor::from_execution_profile(
            StageId::new(2),
            "chat.tensor_static",
            &profile,
            NativeBatchMode::Static,
        );
        tensor.output_visibility = OutputVisibility::IncrementalCommitted;
        assert!(tensor.validate().is_err());
    }

    #[test]
    fn generalized_batch_budget_rejects_overflow_and_excess_work() {
        let budget = BatchBudget {
            max_rows: 2,
            max_logical_units: 8,
            max_tensor_elements: 32,
            max_workspace_bytes: 64,
            max_padding_basis_points: 2_500,
            max_formation_delay: Duration::from_micros(500),
        };
        assert!(budget.validate().is_ok());
        let current = WorkCost::new(3, 12, 24);
        assert!(budget.admits(1, current, WorkCost::new(5, 20, 40)));
        assert!(!budget.admits(1, current, WorkCost::new(6, 20, 40)));
        assert!(!budget.admits(2, current, WorkCost::new(1, 1, 1)));
        assert!(!budget.admits(1, WorkCost::new(u64::MAX, 0, 0), WorkCost::new(1, 0, 0),));
    }

    #[test]
    fn physical_batch_requires_exact_lanes_and_padding_budget() {
        let lane = lane();
        let row = ReadyQuantum {
            plan_id: 1,
            session: SessionKey::new("one".to_string(), 1),
            lane: lane.clone(),
            work: WorkUnit::AtomicJob {
                kind: "test".to_string(),
            },
            cost: WorkCost::new(1, 10, 8),
            managed_cache: None,
        };
        let mut batch = PhysicalBatch {
            batch_id: BatchId::new(1),
            lane: lane.clone(),
            mode: NativeBatchMode::Static,
            budget: BatchBudget {
                max_rows: 2,
                max_logical_units: 2,
                max_tensor_elements: 20,
                max_workspace_bytes: 32,
                max_padding_basis_points: 0,
                max_formation_delay: Duration::ZERO,
            },
            rows: vec![row],
            materialized_tensor_elements: 10,
            workspace: ResourceVector::temporary_workspace(8),
        };
        assert!(batch.validate().is_ok());

        batch.materialized_tensor_elements = 11;
        assert!(batch.validate().is_err());
        batch.materialized_tensor_elements = 10;
        batch.rows[0].lane.shape_bucket = "tokens.2".to_string();
        assert!(batch.validate().is_err());
    }

    #[test]
    fn loaded_model_adapter_and_abi_identity_prevent_cross_cohorting() {
        let baseline = lane();

        let mut reloaded_model = baseline.clone();
        reloaded_model.model_instance = ModelInstanceId::new(99);
        assert_ne!(baseline, reloaded_model);

        let mut reloaded_adapter = baseline.clone();
        reloaded_adapter.adapter_instance = AdapterInstanceId::new(99);
        assert_ne!(baseline, reloaded_adapter);

        let mut upgraded_adapter = baseline.clone();
        upgraded_adapter.adapter_abi = AdapterAbiRevision::new(99);
        assert_ne!(baseline, upgraded_adapter);
    }

    #[test]
    fn physical_batch_reports_are_keyed_instead_of_positional() {
        let lane = lane();
        let mut first = plan_for(
            SessionKey::new("one".to_string(), 1),
            WorkUnit::AtomicJob {
                kind: "test".to_string(),
            },
        );
        first.plan_id = 1;
        first.batch_mode = NativeBatchMode::Static;
        first.max_batch_size = 2;
        let mut second = plan_for(
            SessionKey::new("two".to_string(), 2),
            WorkUnit::AtomicJob {
                kind: "test".to_string(),
            },
        );
        second.plan_id = 2;
        second.batch_mode = NativeBatchMode::Static;
        second.max_batch_size = 2;

        let batch = PhysicalBatch {
            batch_id: BatchId::new(9),
            lane: lane.clone(),
            mode: NativeBatchMode::Static,
            budget: BatchBudget {
                max_rows: 2,
                max_logical_units: 2,
                max_tensor_elements: 20,
                max_workspace_bytes: 32,
                max_padding_basis_points: 0,
                max_formation_delay: Duration::ZERO,
            },
            rows: vec![
                ReadyQuantum {
                    plan_id: first.plan_id,
                    session: first.session.clone(),
                    lane: lane.clone(),
                    work: first.work.clone(),
                    cost: WorkCost::new(1, 10, 8),
                    managed_cache: None,
                },
                ReadyQuantum {
                    plan_id: second.plan_id,
                    session: second.session.clone(),
                    lane: lane.clone(),
                    work: second.work.clone(),
                    cost: WorkCost::new(1, 10, 8),
                    managed_cache: None,
                },
            ],
            materialized_tensor_elements: 20,
            workspace: ResourceVector::temporary_workspace(16),
        };
        let dispatch = BatchDispatch::new(BatchDispatchKind::TensorStatic, 2);
        let mut first_report = report_for(
            &first,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );
        first_report.dispatch = dispatch;
        let mut second_report = report_for(
            &second,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );
        second_report.dispatch = dispatch;
        let active = HashMap::from([(first.plan_id, first), (second.plan_id, second)]);
        let mut report = PhysicalBatchReport {
            batch_id: batch.batch_id,
            lane: lane.clone(),
            dispatch,
            observed_resources: ResourceVector::zero(),
            elapsed: Duration::ZERO,
            rows: vec![
                // Reverse order deliberately: identity, not position, reconciles rows.
                PhysicalBatchRowReport {
                    execution: second_report.clone(),
                    state: StateDisposition::ValidNext,
                    managed_cache: None,
                },
                PhysicalBatchRowReport {
                    execution: first_report.clone(),
                    state: StateDisposition::ValidNext,
                    managed_cache: None,
                },
            ],
        };
        assert!(report.validate_against(&batch, &active).is_ok());

        report.rows[1] = report.rows[0].clone();
        assert!(report.validate_against(&batch, &active).is_err());
        report.rows[1] = PhysicalBatchRowReport {
            execution: first_report,
            state: StateDisposition::ValidNext,
            managed_cache: None,
        };
        report.rows[1].execution.session = SessionKey::new("foreign".to_string(), 99);
        assert!(report.validate_against(&batch, &active).is_err());
    }

    #[test]
    fn request_parallel_report_validates_for_independent_non_tensor_rows() {
        let lane = lane();
        let mut first = plan_for(
            SessionKey::new("parallel-one".to_string(), 1),
            WorkUnit::AtomicJob {
                kind: "test".to_string(),
            },
        );
        first.plan_id = 11;
        first.max_batch_size = 2;
        let mut second = plan_for(
            SessionKey::new("parallel-two".to_string(), 2),
            WorkUnit::AtomicJob {
                kind: "test".to_string(),
            },
        );
        second.plan_id = 12;
        second.max_batch_size = 2;
        let rows = [&first, &second]
            .into_iter()
            .map(|plan| ReadyQuantum {
                plan_id: plan.plan_id,
                session: plan.session.clone(),
                lane: lane.clone(),
                work: plan.work.clone(),
                cost: WorkCost::new(1, 0, 0),
                managed_cache: None,
            })
            .collect::<Vec<_>>();
        let batch = PhysicalBatch {
            batch_id: BatchId::new(10),
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: BatchBudget {
                max_rows: 2,
                max_logical_units: 2,
                max_tensor_elements: u64::MAX,
                max_workspace_bytes: 0,
                max_padding_basis_points: 0,
                max_formation_delay: Duration::ZERO,
            },
            rows,
            materialized_tensor_elements: 0,
            workspace: ResourceVector::zero(),
        };
        let dispatch = BatchDispatch::new(BatchDispatchKind::RequestParallel, 2);
        let report = PhysicalBatchReport {
            batch_id: batch.batch_id,
            lane,
            dispatch,
            observed_resources: ResourceVector::zero(),
            elapsed: Duration::ZERO,
            rows: [&first, &second]
                .into_iter()
                .map(|plan| {
                    let mut execution = report_for(
                        plan,
                        ExecutionDisposition::Finished(FinishReason::Completed),
                    );
                    execution.dispatch = dispatch;
                    PhysicalBatchRowReport {
                        execution,
                        state: StateDisposition::Unchanged,
                        managed_cache: None,
                    }
                })
                .collect(),
        };
        let active = HashMap::from([(first.plan_id, first), (second.plan_id, second)]);

        assert!(batch.validate().is_ok());
        assert!(report.validate_against(&batch, &active).is_ok());
    }

    #[test]
    fn same_session_retry_requires_reusable_model_state() {
        let lane = lane();
        let mut plan = plan_for(
            SessionKey::new("retry".to_string(), 1),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        );
        plan.batch_mode = NativeBatchMode::Continuous;
        let batch = PhysicalBatch {
            batch_id: BatchId::new(10),
            lane: lane.clone(),
            mode: NativeBatchMode::Continuous,
            budget: BatchBudget::width_one(),
            rows: vec![ReadyQuantum {
                plan_id: plan.plan_id,
                session: plan.session.clone(),
                lane: lane.clone(),
                work: plan.work.clone(),
                cost: WorkCost::new(1, 1, 1),
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::temporary_workspace(1),
        };
        let failure = ExecutionFailure {
            kind: FailureKind::Backend,
            scope: FailureScope::Row,
            retry: RetryDisposition::RetrySameSession,
            health: HealthImpact::Degraded,
            message: "retry".to_string(),
        };
        let mut execution = report_for(&plan, ExecutionDisposition::Failed(failure));
        execution.dispatch = BatchDispatch::new(BatchDispatchKind::TensorContinuous, 1);
        let active = HashMap::from([(plan.plan_id, plan)]);
        let mut report = PhysicalBatchReport {
            batch_id: batch.batch_id,
            lane,
            dispatch: execution.dispatch,
            observed_resources: ResourceVector::zero(),
            elapsed: Duration::ZERO,
            rows: vec![PhysicalBatchRowReport {
                execution,
                state: StateDisposition::ValidNext,
                managed_cache: None,
            }],
        };
        assert!(report.validate_against(&batch, &active).is_err());
        report.rows[0].state = StateDisposition::RolledBack;
        assert!(report.validate_against(&batch, &active).is_ok());
    }

    #[test]
    fn managed_cache_report_requires_the_exact_row_receipt() {
        use crate::kv::KvGroupId;

        let lane = lane();
        let session = SessionKey::new("managed".to_string(), 7);
        let plan = plan_for(
            session.clone(),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 4, end: 7 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        );
        let arena = KvArenaId {
            model_instance: lane.model_instance,
            backend: lane.backend,
            device_ordinal: lane.device_ordinal,
            generation: 3,
        };
        let written_blocks = (1..=3)
            .map(|index| CacheBlockRef {
                arena,
                group: KvGroupId::new(0),
                index,
                slot_generation: 8,
            })
            .collect::<Vec<_>>();
        let reservation = ManagedCacheReservation {
            txn_id: plan.plan_id,
            session: session.clone(),
            session_generation: ManagedSessionGeneration::INITIAL,
            domains: vec![ManagedCacheDomainReservation {
                arena,
                domain: CacheDomainId::new(2),
                expected_version: 11,
                expected_committed_tokens: 4,
                execution_start_tokens: 4,
                target_committed_tokens: 7,
                target_window_start: 4,
                first_page_offset: 0,
                provisional_groups: vec![GroupBlockTable {
                    group: KvGroupId::new(0),
                    blocks: written_blocks.clone(),
                }],
                writable_blocks: written_blocks.clone(),
            }],
            clocked_state: None,
        };
        let batch = PhysicalBatch {
            batch_id: BatchId::new(99),
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: BatchBudget::width_one(),
            rows: vec![ReadyQuantum {
                plan_id: plan.plan_id,
                session: session.clone(),
                lane: lane.clone(),
                work: plan.work.clone(),
                cost: WorkCost::new(1, 1, 0),
                managed_cache: Some(reservation.clone()),
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        };
        let mut execution = report_for(&plan, ExecutionDisposition::Progress);
        execution.input_consumed = 1;
        execution.dispatch = BatchDispatch::serial();
        let mut report = PhysicalBatchReport {
            batch_id: batch.batch_id,
            lane,
            dispatch: BatchDispatch::serial(),
            observed_resources: ResourceVector::zero(),
            elapsed: Duration::ZERO,
            rows: vec![PhysicalBatchRowReport {
                execution,
                state: StateDisposition::ValidNext,
                managed_cache: Some(ManagedCacheReceipt {
                    reservation: reservation.clone(),
                    domains: vec![ManagedCacheDomainReceipt {
                        arena,
                        domain: CacheDomainId::new(2),
                        written_blocks: written_blocks.clone(),
                        page_tokens: 1,
                    }],
                    accepted_prefix: None,
                    clocked_state: None,
                }),
            }],
        };
        let active = HashMap::from([(plan.plan_id, plan)]);

        assert!(report.validate_against(&batch, &active).is_ok());
        report.rows[0].managed_cache = Some(
            reservation
                .completed_write_receipt_for_prefix_for_test(5, 1)
                .unwrap(),
        );
        assert!(report.validate_against(&batch, &active).is_ok());
        let mut forged_prefix = report.rows[0].managed_cache.clone().unwrap();
        forged_prefix.accepted_prefix = Some(4);
        report.rows[0].managed_cache = Some(forged_prefix);
        assert!(report.validate_against(&batch, &active).is_err());
        let mut forged_blocks = reservation
            .completed_write_receipt_for_prefix_for_test(6, 1)
            .unwrap();
        forged_blocks.domains[0].written_blocks = vec![written_blocks[2]];
        report.rows[0].managed_cache = Some(forged_blocks);
        assert!(report.validate_against(&batch, &active).is_err());
        report.rows[0].managed_cache = None;
        assert!(report.validate_against(&batch, &active).is_err());
        let mut restart = report_for(
            active.get(&7).unwrap(),
            ExecutionDisposition::RestartSequence(SequenceRestartReason::ModelFallback),
        );
        restart.dispatch = BatchDispatch::serial();
        report.rows[0] = PhysicalBatchRowReport {
            execution: restart,
            state: StateDisposition::RestartPending,
            managed_cache: None,
        };
        assert!(report.validate_against(&batch, &active).is_ok());
        report.rows[0].state = StateDisposition::RolledBack;
        assert!(report.validate_against(&batch, &active).is_err());
        report.rows[0].state = StateDisposition::RestartPending;
        report.rows[0].managed_cache = Some(reservation.completed_write_receipt_for_test());
        assert!(report.validate_against(&batch, &active).is_err());
        let mut foreign = reservation;
        foreign.domains[0].expected_version += 1;
        report.rows[0].managed_cache = Some(ManagedCacheReceipt {
            reservation: foreign,
            domains: Vec::new(),
            accepted_prefix: None,
            clocked_state: None,
        });
        assert!(report.validate_against(&batch, &active).is_err());
    }

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
                auxiliary_state: None,
            },
            batch_key: BatchKey {
                backend: BackendKind::Cpu,
                model_variant: None,
                task_type: TaskType::Chat,
                work_kind: "prefill".to_string(),
                compute_dtype: "f32".to_string(),
                kv_dtype: "f32".to_string(),
                cache_namespace: "none".to_string(),
                adapter: None,
            },
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            estimate: ResourceVector::default(),
            stage: None,
        };
        let report = ExecutionReport {
            plan_id: 7,
            session,
            input_consumed: 5,
            output_produced: 0,
            observed_resources: ResourceVector::default(),
            dispatch: BatchDispatch::serial(),
            provenance: OutcomeProvenance::produced_output(),
            elapsed: Duration::ZERO,
            safe_point: true,
            disposition: ExecutionDisposition::Progress,
            output_finished: false,
            output_has_error: false,
        };
        assert!(report.validate_against(&plan).is_err());
    }

    #[test]
    fn report_authenticates_exact_clocked_state_policy() {
        let selection =
            ClockedStateSelection::new(StateGroupId::new(2), StateClock::AudioSamples).unwrap();
        let span = ClockedStateSpan::new(
            selection.group(),
            selection.clock().clone(),
            InputRange::new(160, 320).unwrap(),
        )
        .unwrap();
        let mut plan = plan_for(
            SessionKey::new("clocked".into(), 1),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange::new(1, 2).unwrap(),
                max_output_steps: 1,
                auxiliary_state: Some(Arc::from([span])),
            },
        );
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "prefill",
            &profile,
            NativeBatchMode::None,
        );
        stage.retained_state_selections = Some(vec![selection]);
        plan.stage = Some(stage);
        let mut report = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );
        report.input_consumed = 1;
        assert!(report.validate_against(&plan).is_ok());

        if let WorkUnit::SequenceStep {
            auxiliary_state, ..
        } = &mut plan.work
        {
            *auxiliary_state = Some(Arc::from([ClockedStateSpan::new(
                StateGroupId::new(2),
                StateClock::AudioFrames,
                InputRange::new(1, 2).unwrap(),
            )
            .unwrap()]));
        }
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
                adapter: None,
            },
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            estimate: ResourceVector::zero(),
            stage: None,
        }
    }

    fn report_for(plan: &ExecutionPlan, disposition: ExecutionDisposition) -> ExecutionReport {
        let (output_finished, output_has_error) = match &disposition {
            ExecutionDisposition::Progress
            | ExecutionDisposition::Yielded(_)
            | ExecutionDisposition::RestartSequence(_) => (false, false),
            ExecutionDisposition::Finished(_) => (true, false),
            ExecutionDisposition::Failed(failure) => {
                (failure.retry == RetryDisposition::Never, true)
            }
        };
        let provenance = match &disposition {
            ExecutionDisposition::Failed(_) => {
                OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
            }
            ExecutionDisposition::Finished(FinishReason::TimedOut) => {
                OutcomeProvenance::deadline(DeadlinePhase::ModelExecution, DispatchState::Started)
            }
            _ => OutcomeProvenance::produced_output(),
        };
        ExecutionReport {
            plan_id: plan.plan_id,
            session: plan.session.clone(),
            input_consumed: 0,
            output_produced: 0,
            observed_resources: ResourceVector::zero(),
            dispatch: BatchDispatch::serial(),
            provenance,
            elapsed: Duration::ZERO,
            safe_point: true,
            disposition,
            output_finished,
            output_has_error,
        }
    }

    #[test]
    fn pre_sequence_preparation_cannot_terminalize_an_engine_core_session() {
        let plan = plan_for(
            SessionKey::new("pre-core-preparation".to_string(), 1),
            WorkUnit::PreSequencePreparation {
                kind: "asr.encoder.audio".to_string(),
            },
        );
        let report = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );

        let error = report
            .validate_against(&plan)
            .expect_err("pre-sequence preparation must remain outside EngineCore");
        assert!(error.to_string().contains("before EngineCore admission"));
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
                auxiliary_state: None,
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
                auxiliary_state: None,
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
    fn sequence_restart_requires_zero_progress_safe_point_and_enters_restart_pending() {
        let session = SessionKey::new("restart".to_string(), 9);
        let plan = plan_for(
            session.clone(),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 4, end: 5 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        );
        let disposition =
            ExecutionDisposition::RestartSequence(SequenceRestartReason::ModelFallback);
        let report = report_for(&plan, disposition.clone());
        assert!(report.validate_against(&plan).is_ok());

        let mut unsafe_report = report.clone();
        unsafe_report.safe_point = false;
        assert!(unsafe_report.validate_against(&plan).is_err());
        let mut progressed = report.clone();
        progressed.output_produced = 1;
        assert!(progressed.validate_against(&plan).is_err());

        let mut tracker = ExecutionTracker::new(session);
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::Decoding).unwrap();
        tracker.begin_plan(&plan).unwrap();
        tracker.commit(&plan, &report).unwrap();
        assert_eq!(tracker.state(), ExecutionState::RestartPending);
        assert_eq!(tracker.active_plan_id(), None);
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
                auxiliary_state: None,
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
                auxiliary_state: None,
            },
        );
        let mut tracker = ExecutionTracker::new(session);
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::Decoding).unwrap();
        tracker.begin_plan(&plan).unwrap();
        let retryable = ExecutionFailure {
            kind: FailureKind::Backend,
            scope: FailureScope::Row,
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
                auxiliary_state: None,
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
            scope: FailureScope::Row,
            retry: RetryDisposition::RetrySameSession,
            health: HealthImpact::Degraded,
            message: "transient".to_string(),
        };
        let mut retryable = report_for(&plan, ExecutionDisposition::Failed(retry));
        retryable.output_finished = true;
        assert!(retryable.validate_against(&plan).is_err());
    }

    #[test]
    fn provenance_must_match_dispatch_failure_and_deadline_outcomes() {
        let plan = plan_for(
            SessionKey::new("provenance".to_string(), 1),
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 0 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        );
        let mut failed = report_for(
            &plan,
            ExecutionDisposition::Failed(ExecutionFailure::invalid_output("failed")),
        );
        failed.dispatch = BatchDispatch::not_dispatched(1);
        assert!(failed.validate_against(&plan).is_err());
        failed.provenance = OutcomeProvenance::failure(
            FailureOrigin::ExecutorValidation,
            DispatchState::NotStarted,
        );
        assert!(failed.validate_against(&plan).is_ok());

        let mut timed_out = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::TimedOut),
        );
        assert!(timed_out.validate_against(&plan).is_ok());
        timed_out.provenance = OutcomeProvenance::started();
        assert!(timed_out.validate_against(&plan).is_err());

        let mut completed = report_for(
            &plan,
            ExecutionDisposition::Finished(FinishReason::Completed),
        );
        completed.provenance = OutcomeProvenance::deadline(
            DeadlinePhase::ModelExecution,
            DispatchState::ProducedOutput,
        );
        assert!(completed.validate_against(&plan).is_err());
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
                auxiliary_state: None,
            },
        );
        let mut tracker = ExecutionTracker::new(session);
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::Decoding).unwrap();
        tracker.begin_plan(&plan).unwrap();

        let recompute = ExecutionFailure {
            kind: FailureKind::Backend,
            scope: FailureScope::Row,
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
        assert_eq!(
            profile.physical_launch_policy,
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
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
        profile.cache_mode = CacheMode::None;
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

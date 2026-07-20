//! Authoritative execution plans, reports, capabilities, and lifecycle states.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::backends::BackendKind;
use crate::error::{Error, Result};
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
    SequencePrefill,
    SequenceDecode,
    Atomic,
    Pipeline { ordinal: Option<usize> },
}

impl StageWorkSelector {
    fn matches(self, work: &WorkUnit) -> bool {
        match (self, work) {
            (Self::Any, _) => true,
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
    AfterQuantumCommit,
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
        let shape_policy = match batch_mode {
            NativeBatchMode::None => StageShapePolicy::Exact,
            NativeBatchMode::Static => StageShapePolicy::Padded,
            NativeBatchMode::Continuous => StageShapePolicy::Ragged,
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
            batch_mode,
            max_batch_size: if batch_mode == NativeBatchMode::None {
                1
            } else {
                profile.max_batch_size.max(1)
            },
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
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::InvalidInput(
                "execution stage name cannot be empty".to_string(),
            ));
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
        if self.batch_mode == NativeBatchMode::None && self.max_batch_size != 1 {
            return Err(Error::InvalidInput(
                "non-batchable execution stages must have width one".to_string(),
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkCost {
    pub logical_units: u64,
    pub tensor_elements: u64,
    pub workspace_bytes: u64,
}

impl WorkCost {
    pub const fn new(logical_units: u64, tensor_elements: u64, workspace_bytes: u64) -> Self {
        Self {
            logical_units,
            tensor_elements,
            workspace_bytes,
        }
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            logical_units: self.logical_units.checked_add(other.logical_units)?,
            tensor_elements: self.tensor_elements.checked_add(other.tensor_elements)?,
            workspace_bytes: self.workspace_bytes.checked_add(other.workspace_bytes)?,
        })
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
            && total.workspace_bytes <= self.max_workspace_bytes
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
    pub workspace_bytes: u64,
}

impl PhysicalBatch {
    pub fn validate(&self) -> Result<()> {
        self.budget.validate()?;
        if self.rows.is_empty() {
            return Err(Error::InvalidInput(
                "physical batch cannot be empty".to_string(),
            ));
        }
        if self.mode == NativeBatchMode::None && self.rows.len() != 1 {
            return Err(Error::InvalidInput(
                "non-tensor physical dispatch must have width one".to_string(),
            ));
        }

        let mut keys = HashSet::with_capacity(self.rows.len());
        let mut cost = WorkCost::default();
        let mut row_count = 0usize;
        for row in &self.rows {
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
            row_count += 1;
        }

        if self.materialized_tensor_elements < cost.tensor_elements {
            return Err(Error::InvalidInput(
                "physical batch materialization is smaller than useful tensor work".to_string(),
            ));
        }
        if self.workspace_bytes > self.budget.max_workspace_bytes {
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
    Poisoned,
}

#[derive(Debug, Clone)]
pub struct PhysicalBatchRowReport {
    pub execution: ExecutionReport,
    pub state: StateDisposition,
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
        match self.dispatch.kind {
            BatchDispatchKind::NotDispatched
                if self.rows.iter().any(|row| {
                    !matches!(row.execution.disposition, ExecutionDisposition::Failed(_))
                }) =>
            {
                return Err(Error::InferenceError(
                    "a non-dispatched batch may only report failed rows".to_string(),
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
            BatchDispatchKind::RequestParallel => {
                return Err(Error::InferenceError(
                    "request parallelism is not a physical tensor batch".to_string(),
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
            BatchDispatchKind::NotDispatched
                if !matches!(self.disposition, ExecutionDisposition::Failed(_)) =>
            {
                return Err(Error::InferenceError(
                    "non-dispatched execution must report failure".to_string(),
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
        assert!(stage.validate().is_ok());
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
        };
        let decode_work = WorkUnit::SequenceStep {
            phase: SequencePhase::Decode,
            input: InputRange { start: 8, end: 9 },
            max_output_steps: 1,
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
    fn continuous_stage_requires_repeatable_safe_points() {
        let invalid = StageDescriptor {
            id: StageId::new(2),
            name: "atomic".to_string(),
            selector: StageWorkSelector::Atomic,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
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
            workspace_bytes: 8,
        };
        assert!(batch.validate().is_ok());

        batch.materialized_tensor_elements = 11;
        assert!(batch.validate().is_err());
        batch.materialized_tensor_elements = 10;
        batch.rows[0].lane.shape_bucket = "tokens.2".to_string();
        assert!(batch.validate().is_err());
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
                },
                ReadyQuantum {
                    plan_id: second.plan_id,
                    session: second.session.clone(),
                    lane: lane.clone(),
                    work: second.work.clone(),
                    cost: WorkCost::new(1, 10, 8),
                },
            ],
            materialized_tensor_elements: 20,
            workspace_bytes: 16,
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
                },
                PhysicalBatchRowReport {
                    execution: first_report.clone(),
                    state: StateDisposition::ValidNext,
                },
            ],
        };
        assert!(report.validate_against(&batch, &active).is_ok());

        report.rows[1] = report.rows[0].clone();
        assert!(report.validate_against(&batch, &active).is_err());
        report.rows[1] = PhysicalBatchRowReport {
            execution: first_report,
            state: StateDisposition::ValidNext,
        };
        report.rows[1].execution.session = SessionKey::new("foreign".to_string(), 99);
        assert!(report.validate_against(&batch, &active).is_err());
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
            }],
            materialized_tensor_elements: 1,
            workspace_bytes: 1,
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
            }],
        };
        assert!(report.validate_against(&batch, &active).is_err());
        report.rows[0].state = StateDisposition::RolledBack;
        assert!(report.validate_against(&batch, &active).is_ok());
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

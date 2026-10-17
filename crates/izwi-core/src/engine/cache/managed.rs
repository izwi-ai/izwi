//! Live binding between loaded-model KV contracts and physical engine state.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, DeviceLocation};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::coordinator::{
    KvBlockIntent, KvCacheCoordinator, KvCoordinatorCommitPlan, KvCoordinatorError,
    KvCoordinatorTableResetPlan, KvGroupReservation, KvReserveRequest, KvSnapshot,
    KvWindowReserveRequest, KvWriteReceipt,
};
use super::offload::{self, DemoteOutcome, HostOffloadPolicy};
use super::prefix::{
    CoordinatedPrefixIndex, KvHostTailMatch, KvPrefixMatch, KvPrefixNamespace, KvPrefixPageKey,
    KvPrefixPublication, StagedPrefixCommit,
};
use super::telemetry::{ManagedKvTelemetry, ManagedKvTelemetrySnapshot};
use super::tensor_snapshots::{
    TensorSnapshotPrefixPolicy, TensorStateSnapshotIndex, tensor_snapshot_backing_bytes,
    tensor_snapshot_prefix_policy,
};
use crate::backends::{BackendKind, DeviceThread};
#[cfg(feature = "cuda")]
use crate::backends::kv::CudaKvBackendRuntime;
#[cfg(feature = "metal")]
use crate::backends::kv::MetalKvBackendRuntime;
use crate::backends::kv::{
    CpuKvBackendRuntime, KvArena, KvArenaConfig, KvArenaGrowthConfig, KvBackendRuntime,
    KvLayerConfig, cuda_paged_growth_geometry,
};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateBackendPlanRequest,
    StateBackendRegistry, StateComponentValue, TensorStateArena, TensorStateCapacity,
    TensorStateSelection, negotiate_state_plan,
};
use crate::engine::{
    EngineCoreRequest, ManagedCacheDomainReservation, ManagedCacheReceipt, ManagedCacheReservation,
    ManagedClockedStateReservation, ManagedSessionGeneration, ModelInstanceId, PlanId,
    ReservationClass, ReservationOwner, ResourceAmount, ResourceAuthority, ResourceLease,
    ResourceVector, SessionKey, WorkUnit,
};
use crate::error::{Error, Result};
#[cfg(test)]
use crate::kv::CacheDomainId;
use crate::kv::v2::{
    AllocationReceipt, AttentionPattern, CapacityStrategy, GroupCapacityRequest,
    InferenceStateContract, PrefixPolicy, ResidencyMeasurement, ResolvedStatePlan,
    StateAllocationLedger, StateDomainId, StateDomainSpec, StateResourceVector,
    StateRuntimeAllocationPlan, WorkspaceContract, WorkspacePlacement,
};
use crate::kv::{InferenceStateCapability, KvArenaId, KvGroupId, KvStorageDType, ResolvedKvPlan};

/// Default snapshot budget of a hybrid model that shares committed prefix
/// snapshots, in rows of its recurrent state: one continuing conversation per
/// row of a full decode step (8 rows), each attaching the snapshot its
/// previous turn published. 8 x 156,893,184 B = 1.26 GB for Qwen3.8-27B.
pub(crate) const DEFAULT_PREFIX_SNAPSHOT_ROWS: u32 = 8;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvOperationSnapshot {
    pub slot_write_dispatches: u64,
    pub paged_prefill_dispatches: u64,
    pub paged_decode_dispatches: u64,
    pub page_zero_dispatches: u64,
    pub page_copy_dispatches: u64,
    /// Long-lived K/V backing allocations reported by metered arenas.
    pub backing_allocations: u64,
    pub backing_allocations_observed_arenas: u64,
    /// Retained provider workspace reported by metered arenas.
    pub workspace_bytes: u64,
    pub workspace_bytes_observed_arenas: u64,
    pub workspace_budget_bytes: u64,
    pub workspace_budget_bytes_observed_arenas: u64,
    pub workspace_high_water_bytes: u64,
    pub workspace_high_water_bytes_observed_arenas: u64,
    pub workspace_allocations: u64,
    pub workspace_allocations_observed_arenas: u64,
    pub host_synchronizations: u64,
    pub cpu_reference_attention_dispatches: u64,
    pub portable_attention_dispatches: u64,
    pub cuda_native_attention_dispatches: u64,
    pub cuda_flash_attention_dispatches: u64,
    pub cuda_split_decode_dispatches: u64,
    pub metal_native_attention_dispatches: u64,
    pub cuda_graph_warmups: u64,
    pub cuda_graph_captures: u64,
    pub cuda_graph_replays: u64,
    pub cuda_graph_fallbacks: u64,
    pub cuda_graph_backoff_hits: u64,
    pub cuda_graph_evictions: u64,
}

impl ManagedKvOperationSnapshot {
    fn add_assign(&mut self, other: Self) {
        self.slot_write_dispatches = self
            .slot_write_dispatches
            .saturating_add(other.slot_write_dispatches);
        self.paged_decode_dispatches = self
            .paged_decode_dispatches
            .saturating_add(other.paged_decode_dispatches);
        self.paged_prefill_dispatches = self
            .paged_prefill_dispatches
            .saturating_add(other.paged_prefill_dispatches);
        self.page_zero_dispatches = self
            .page_zero_dispatches
            .saturating_add(other.page_zero_dispatches);
        self.page_copy_dispatches = self
            .page_copy_dispatches
            .saturating_add(other.page_copy_dispatches);
        self.backing_allocations = self
            .backing_allocations
            .saturating_add(other.backing_allocations);
        self.backing_allocations_observed_arenas = self
            .backing_allocations_observed_arenas
            .saturating_add(other.backing_allocations_observed_arenas);
        self.workspace_bytes = self.workspace_bytes.saturating_add(other.workspace_bytes);
        self.workspace_bytes_observed_arenas = self
            .workspace_bytes_observed_arenas
            .saturating_add(other.workspace_bytes_observed_arenas);
        self.workspace_budget_bytes = self
            .workspace_budget_bytes
            .saturating_add(other.workspace_budget_bytes);
        self.workspace_budget_bytes_observed_arenas = self
            .workspace_budget_bytes_observed_arenas
            .saturating_add(other.workspace_budget_bytes_observed_arenas);
        self.workspace_high_water_bytes = self
            .workspace_high_water_bytes
            .saturating_add(other.workspace_high_water_bytes);
        self.workspace_high_water_bytes_observed_arenas = self
            .workspace_high_water_bytes_observed_arenas
            .saturating_add(other.workspace_high_water_bytes_observed_arenas);
        self.workspace_allocations = self
            .workspace_allocations
            .saturating_add(other.workspace_allocations);
        self.workspace_allocations_observed_arenas = self
            .workspace_allocations_observed_arenas
            .saturating_add(other.workspace_allocations_observed_arenas);
        self.host_synchronizations = self
            .host_synchronizations
            .saturating_add(other.host_synchronizations);
        self.cpu_reference_attention_dispatches = self
            .cpu_reference_attention_dispatches
            .saturating_add(other.cpu_reference_attention_dispatches);
        self.portable_attention_dispatches = self
            .portable_attention_dispatches
            .saturating_add(other.portable_attention_dispatches);
        self.cuda_native_attention_dispatches = self
            .cuda_native_attention_dispatches
            .saturating_add(other.cuda_native_attention_dispatches);
        self.cuda_flash_attention_dispatches = self
            .cuda_flash_attention_dispatches
            .saturating_add(other.cuda_flash_attention_dispatches);
        self.cuda_split_decode_dispatches = self
            .cuda_split_decode_dispatches
            .saturating_add(other.cuda_split_decode_dispatches);
        self.metal_native_attention_dispatches = self
            .metal_native_attention_dispatches
            .saturating_add(other.metal_native_attention_dispatches);
        self.cuda_graph_warmups = self
            .cuda_graph_warmups
            .saturating_add(other.cuda_graph_warmups);
        self.cuda_graph_captures = self
            .cuda_graph_captures
            .saturating_add(other.cuda_graph_captures);
        self.cuda_graph_replays = self
            .cuda_graph_replays
            .saturating_add(other.cuda_graph_replays);
        self.cuda_graph_fallbacks = self
            .cuda_graph_fallbacks
            .saturating_add(other.cuda_graph_fallbacks);
        self.cuda_graph_backoff_hits = self
            .cuda_graph_backoff_hits
            .saturating_add(other.cuda_graph_backoff_hits);
        self.cuda_graph_evictions = self
            .cuda_graph_evictions
            .saturating_add(other.cuda_graph_evictions);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvCoordinatorSnapshot {
    pub capacity_pages: u64,
    pub allocated_pages: u64,
    pub free_pages: u64,
    /// Pages promised to active requests by conservative or incremental
    /// admission. This is logical admission, not eager backing.
    pub admission_claimed_pages: u64,
    pub admission_available_pages: u64,
    pub admission_claims: u64,
    pub table_refs: u64,
    pub prefix_refs: u64,
    pub execution_pins: u64,
    pub transfer_pins: u64,
    pub reservations: u64,
    pub active_transactions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagedKvArenaRuntimeSnapshot {
    pub generation: u32,
    pub group_id: u32,
    pub domain_id: u32,
    pub device_ordinal: Option<u32>,
    pub page_tokens: u32,
    /// Resolved page storage dtype (`f32`, `f16`, `bf16`).
    pub storage_dtype: String,
    pub token_capacity: u64,
    pub bytes_per_page: u64,
    pub physical_bytes: u64,
    /// Pages the arena's committed-prefix index may retain: the effective
    /// prefix page budget for this model (see `prefix_index_pages`).
    pub prefix_capacity_pages: u64,
    pub coordinator: ManagedKvCoordinatorSnapshot,
    pub operations: ManagedKvOperationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagedKvModelRuntimeSnapshot {
    pub model_instance: ModelInstanceId,
    pub plan_fingerprint: String,
    pub state_plan_v2_fingerprint: String,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    /// Resident paged-attention backing observed from the backend arena.
    pub resident_paged_bytes: u64,
    /// Maximum retained tensor-state envelope authorized for committed and
    /// staged rows. Tensor values are materialized on demand.
    pub authorized_tensor_bytes: u64,
    /// Committed tensor-state backing recorded as materialized on the model
    /// lease; the rest of `authorized_tensor_bytes` is a pending claim.
    pub materialized_tensor_bytes: u64,
    /// Compatibility total: resident paged bytes plus authorized tensor bytes.
    pub physical_bytes: u64,
    pub registered_sessions: u64,
    /// Maximum logical token reach of one sequence across every state arena.
    pub single_sequence_token_capacity: u64,
    /// Aggregate token capacity of the smallest shared paged arena.
    pub aggregate_token_capacity: u64,
    /// Number of maximum-reach sequences the fitted page pools can admit.
    pub full_context_sequence_capacity: u64,
    /// Retained sequence rows backed by recurrent tensor state. `None` for
    /// paged-only plans, whose sequences are bounded by pages alone.
    pub retained_sequence_capacity: Option<u32>,
    /// Recurrent tensor-state rows that may be staged at once.
    pub staged_transaction_capacity: Option<u32>,
    /// Recurrent tensor state of one retained or staged row.
    pub tensor_bytes_per_sequence: Option<u64>,
    /// Committed prefix snapshots the shared-prefix index may hold (each one
    /// row of hybrid state), reserved on the model lease beside the rows.
    /// `None` for plans without tensor state; zero when snapshots do not share.
    pub prefix_snapshot_capacity: Option<u32>,
    /// Byte budget of those snapshots (`prefix_snapshot_capacity` rows).
    pub prefix_snapshot_budget_bytes: u64,
    /// Snapshots the index holds now, and the backing bytes they keep alive
    /// (counted in `materialized_tensor_bytes`, once per allocation).
    pub prefix_snapshots: u64,
    pub prefix_snapshot_bytes: u64,
    /// Active sessions admitted with bounded next-step claims.
    pub incremental_claim_sessions: u64,
    pub arenas: Vec<ManagedKvArenaRuntimeSnapshot>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvRuntimeTotalsSnapshot {
    pub models: u64,
    pub arenas: u64,
    pub registered_sessions: u64,
    pub resident_paged_bytes: u64,
    pub authorized_tensor_bytes: u64,
    pub materialized_tensor_bytes: u64,
    /// Compatibility total: resident paged bytes plus authorized tensor bytes.
    pub physical_bytes: u64,
    pub coordinator: ManagedKvCoordinatorSnapshot,
    pub operations: ManagedKvOperationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManagedKvRuntimeSnapshot {
    pub memory_accounting: &'static str,
    pub totals: ManagedKvRuntimeTotalsSnapshot,
    pub counters: ManagedKvTelemetrySnapshot,
    pub models: Vec<ManagedKvModelRuntimeSnapshot>,
}

impl Default for ManagedKvRuntimeSnapshot {
    fn default() -> Self {
        Self {
            memory_accounting: "resident_paged_plus_authorized_tensor",
            totals: ManagedKvRuntimeTotalsSnapshot::default(),
            counters: ManagedKvTelemetrySnapshot::default(),
            models: Vec::new(),
        }
    }
}

/// Immutable model-level plan and physical arenas shared by all its sessions.
pub(crate) struct ManagedKvModelRuntime {
    plan: Arc<ResolvedKvPlan>,
    state_plan_v2: Arc<ResolvedStatePlan>,
    allocation_plan: Arc<StateRuntimeAllocationPlan>,
    arenas: HashMap<KvArenaId, Arc<dyn KvArena>>,
    tensor_state: Option<Arc<TensorStateArena>>,
    non_paged_physical_bytes: u64,
    maximum_sequence_tokens: AtomicU64,
}

impl fmt::Debug for ManagedKvModelRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedKvModelRuntime")
            .field("plan", &self.plan.id)
            .field("state_plan_v2", &self.state_plan_v2.id)
            .field("allocation_plan", &self.allocation_plan.id)
            .field("arena_count", &self.arenas.len())
            .field("physical_bytes", &self.physical_bytes())
            .finish()
    }
}

impl ManagedKvModelRuntime {
    pub(crate) fn plan(&self) -> &ResolvedKvPlan {
        &self.plan
    }

    pub(crate) fn state_plan_v2(&self) -> &ResolvedStatePlan {
        &self.state_plan_v2
    }

    pub(crate) fn allocation_plan(&self) -> &StateRuntimeAllocationPlan {
        &self.allocation_plan
    }

    pub(crate) fn logical_token_reach(&self) -> u64 {
        self.plan
            .groups
            .iter()
            .map(|group| u64::from(group.capacity_pages) * u64::from(group.page_tokens))
            .min()
            .unwrap_or(0)
    }

    /// Model context and aggregate page capacity are independent controls.
    /// Called during model publication, before requests can observe the runtime.
    pub(crate) fn set_maximum_sequence_tokens(&self, tokens: u64) {
        self.maximum_sequence_tokens
            .store(tokens, Ordering::Relaxed);
    }

    pub(crate) fn maximum_sequence_tokens(&self) -> u64 {
        let configured = self.maximum_sequence_tokens.load(Ordering::Relaxed);
        if configured == 0 {
            self.logical_token_reach()
        } else {
            configured
        }
    }

    /// Make asynchronous accelerator allocation/zeroing failures observable
    /// before the lifecycle publishes this generation as Ready.
    pub(crate) fn synchronize_backing(&self) -> Result<()> {
        for arena in self.arenas.values() {
            arena.drain()?;
        }
        Ok(())
    }

    pub(crate) fn arena(&self, id: KvArenaId) -> Option<&Arc<dyn KvArena>> {
        self.arenas.get(&id)
    }

    pub(crate) fn tensor_state(&self) -> Option<&Arc<TensorStateArena>> {
        self.tensor_state.as_ref()
    }

    pub(crate) fn physical_bytes(&self) -> u64 {
        self.resident_paged_bytes()
            .saturating_add(self.authorized_tensor_bytes())
    }

    pub(crate) fn resident_paged_bytes(&self) -> u64 {
        self.arenas.values().fold(0_u64, |total, arena| {
            total.saturating_add(arena.resident_bytes())
        })
    }

    pub(crate) const fn authorized_tensor_bytes(&self) -> u64 {
        self.non_paged_physical_bytes
    }
}

struct ManagedKvModelState {
    closing: bool,
    contract: InferenceStateContract,
    runtime: Arc<ManagedKvModelRuntime>,
    coordinators: HashMap<KvArenaId, KvCacheCoordinator>,
    prefix_indexes: HashMap<KvArenaId, CoordinatedPrefixIndex>,
    pending_prefixes: HashMap<PlanId, Vec<PendingPrefixCommit>>,
    /// DS4.3: promotions awaiting their transaction's commit, keyed by txn.
    pending_promotions: HashMap<PlanId, Vec<PendingPrefixPromotion>>,
    /// Transactions whose accepted prefix must equal their reserved target.
    exact_target_transactions: HashSet<PlanId>,
    registered_sessions: HashSet<SessionKey>,
    session_generations: HashMap<SessionKey, ManagedSessionGeneration>,
    capacity_claims: HashMap<SessionKey, Vec<(KvArenaId, u32)>>,
    incremental_claim_sessions: HashSet<SessionKey>,
    tensor_sequences: HashMap<SessionKey, PhysicalStateSequenceId>,
    tensor_snapshots: Option<TensorSnapshotSharing>,
    /// DS4 host-tier state: one bounded pool + chain index per arena when
    /// hierarchical offload is engaged for this engine.
    host_offload: Option<offload::ModelOffload>,
    resource_lease: Option<ResourceLease>,
    /// Paged backing (and the host tier's budget) recorded on the lease.
    materialized_resources: ResourceVector,
    /// Tensor-arena committed backing recorded on the lease beside
    /// `materialized_resources`; the rest of the tensor envelope is pending.
    tensor_materialized_bytes: u64,
    /// The requests that create this model's tensor rows already reserve
    /// them, so the whole envelope is recorded on the lease at load and is
    /// never a pending claim (see
    /// [`ManagedKvCacheManager::price_tensor_rows_with_requests`]).
    tensor_rows_request_priced: bool,
    allocation_ledger: StateAllocationLedger,
}

#[derive(Clone)]
struct PendingPrefixCommit {
    arena: KvArenaId,
    page_tokens: u32,
    publications: Vec<KvPrefixPublication>,
    /// Whether this span may publish the committed tensor snapshot at its
    /// commit cursor (DS1.2b). A request declaring a snapshot boundary
    /// (Phase 5a) snapshots only at that boundary.
    snapshot: bool,
}

/// DS4.3: host-resident pages a prepare restored into its transaction's fresh
/// device pages. At commit the device pages publish into the prefix index and
/// the host entries are reclaimed; on abort the record drops and the host
/// entries stay valid. `blocks` are the device pages the bytes were restored
/// into, `digests` the host identities they came from, all in chain
/// order starting at the device match boundary.
#[derive(Clone)]
struct PendingPrefixPromotion {
    arena: KvArenaId,
    page_tokens: u32,
    device_end_tokens: u32,
    blocks: Vec<crate::kv::CacheBlockRef>,
    digests: Vec<[u8; 32]>,
    latency_ns: u64,
}

/// DS1.2b: cross-request sharing of committed tensor snapshots for one hybrid
/// contract. The snapshot key is the paged page-chain digest whose page ends
/// at the snapshot cursor, so tensor reuse can never exceed or diverge from
/// the paged reuse the same lookup just authenticated.
struct TensorSnapshotSharing {
    policy: TensorSnapshotPrefixPolicy,
    /// The paged group's arena whose prefix index publishes that chain.
    paged_arena: KvArenaId,
    /// Resolved page tokens of the paged group; the declared snapshot interval
    /// must be a multiple of it so boundaries are also page boundaries.
    interval_tokens: u32,
    index: TensorStateSnapshotIndex,
}

/// Engine-owned managed-cache registry. Arena backing is allocated once per
/// exact model instance; row transactions only change page ownership.
pub(crate) struct ManagedKvCacheManager {
    models: HashMap<ModelInstanceId, ManagedKvModelState>,
    resource_authority: Option<Arc<ResourceAuthority>>,
    next_arena_generation: u32,
    next_tensor_sequence: u64,
    /// Fork-seed transaction ids descend from the top of the id space so they
    /// never collide with plan-id transaction ids that ascend from zero.
    next_tensor_fork_transaction: u64,
    telemetry: Arc<ManagedKvTelemetry>,
    prefix_cache_salt: Option<[u8; 32]>,
    max_prefix_cache_pages: usize,
    /// The operator left the prefix page budget unset, so arenas that the
    /// CUDA resident fit allocates at load size theirs from the arena (see
    /// [`prefix_index_pages`]).
    automatic_prefix_cache_pages: bool,
    /// Committed hybrid-state prefix snapshots a snapshot-sharing model may
    /// hold beyond its live rows (see [`Self::prefix_snapshot_rows_for`]).
    prefix_snapshot_rows: u32,
    host_offload_policy: Option<offload::HostOffloadPolicy>,
    worker_backend: BackendKind,
    worker_device_location: DeviceLocation,
    worker_device: Device,
    worker_device_ordinal: Option<u32>,
    backend_runtime: Option<Arc<dyn KvBackendRuntime>>,
    backend_unavailable: Option<String>,
    #[cfg(test)]
    fail_next_composite_synchronize: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ManagedStateCapacityRequest {
    /// Aggregate backing pages shared fairly across every paged state group.
    pub(crate) total_paged_pages: u32,
    /// Exact logical token reach carried by a loaded CUDA model. Portable
    /// backends leave this unset and retain the configured aggregate budget.
    pub(crate) logical_token_reach: Option<u64>,
    /// Maximum number of retained sequences that may remain registered while
    /// the scheduler interleaves their execution.
    pub(crate) retained_sequence_rows: u32,
    /// Maximum number of retained-state transactions that may be staged at
    /// once. This is an independent axis even when the current scheduler sets
    /// it equal to `retained_sequence_rows`.
    pub(crate) staged_transaction_rows: u32,
}

impl Default for ManagedKvCacheManager {
    fn default() -> Self {
        Self::new(None)
    }
}

/// What a portable managed-state fit may spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PortableFitHeadroom {
    /// Stable planning headroom (chat and Fish state). `minimum_tokens` is
    /// the context the model needs to be useful (chat: one ordinary
    /// conversation; `0`: the page minimum). A load that cannot reach it is
    /// refused with a capacity error instead of publishing a tiny context.
    Planning { minimum_tokens: u64 },
    /// Planning headroom clamped to live memory
    /// ([`ResourceAuthority::state_fit_headroom_bytes`]): ASR and TTS
    /// retained state, which must leave room for the requests it serves.
    /// `minimum_tokens` is the floor each sequence must reach (one ASR chunk,
    /// or one TTS prompt plus its output budget; `0`: the page minimum); it
    /// is admitted against planning headroom or the load is refused.
    Live {
        minimum_tokens: u64,
        reserve: StateFitReserve,
    },
}

/// The safety reserve a live state fit leaves free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateFitReserve {
    /// The configured floor or the measured untracked device memory,
    /// whichever is larger (ADM-11). Right for a load whose weight growth
    /// was observed and reconciled: what remains untracked is pool slack and
    /// other growth that recurs beside the state.
    MeasuredUntracked,
    /// The configured floor alone, for a load without a device observation.
    /// Its weights are priced from the checkpoint, not observed, so on Metal
    /// the whole under-pricing (Qwen3-TTS 0.6B: about 5.2 GB held against
    /// 2.5 GB charged) reads as untracked memory. The live sample already
    /// charges those bytes, so reserving them again left no growth budget
    /// and collapsed TTS state to one page (R-1).
    Configured,
}

impl StateFitReserve {
    /// The reserve for a load that did (`true`) or did not observe its
    /// weight growth.
    pub(crate) const fn for_observed_load(observed: bool) -> Self {
        if observed {
            Self::MeasuredUntracked
        } else {
            Self::Configured
        }
    }
}

impl ManagedKvCacheManager {
    pub(crate) fn contains_model(&self, model_instance: ModelInstanceId) -> bool {
        self.models.contains_key(&model_instance)
    }

    pub(crate) fn synchronize_worker(&self) -> Result<()> {
        self.worker_device.synchronize().map_err(Error::from)
    }
    #[cfg(test)]
    pub(crate) fn inject_composite_synchronize_failure(&mut self) {
        self.fail_next_composite_synchronize = true;
    }

    #[cfg(test)]
    pub(crate) fn take_composite_synchronize_failure(&mut self) -> bool {
        std::mem::take(&mut self.fail_next_composite_synchronize)
    }
    #[cfg(test)]
    pub(crate) fn model_count(&self) -> usize {
        self.models.len()
    }

    pub(crate) fn new(resource_authority: Option<Arc<ResourceAuthority>>) -> Self {
        Self::for_worker(resource_authority, BackendKind::Cpu, Device::Cpu)
    }

    pub(crate) fn for_worker(
        resource_authority: Option<Arc<ResourceAuthority>>,
        backend: BackendKind,
        device: Device,
    ) -> Self {
        let (backend_runtime, backend_unavailable) = managed_backend_runtime(backend, &device);
        Self {
            models: HashMap::new(),
            resource_authority,
            next_arena_generation: 1,
            next_tensor_sequence: 1,
            next_tensor_fork_transaction: 0,
            telemetry: Arc::new(ManagedKvTelemetry::default()),
            prefix_cache_salt: None,
            max_prefix_cache_pages: 0,
            automatic_prefix_cache_pages: false,
            prefix_snapshot_rows: DEFAULT_PREFIX_SNAPSHOT_ROWS,
            host_offload_policy: None,
            worker_backend: backend,
            worker_device_location: device.location(),
            worker_device: device.clone(),
            worker_device_ordinal: managed_device_ordinal(&device),
            backend_runtime,
            backend_unavailable,
            #[cfg(test)]
            fail_next_composite_synchronize: false,
        }
    }

    /// Resolve the largest portable logical reach whose exact managed-state
    /// plan fits the stable shared-memory envelope. CUDA deliberately bypasses
    /// this path and keeps admission-growable arenas.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit_portable_logical_token_reach(
        &self,
        model_instance: ModelInstanceId,
        contract: &InferenceStateContract,
        maximum_tokens: u64,
        safety_reserve_bytes: u64,
        page_tokens_hint: usize,
        retained_sequence_rows: u32,
        staged_transaction_rows: u32,
        portable_state_copies: u32,
        headroom: PortableFitHeadroom,
    ) -> Result<u64> {
        if self.worker_backend == BackendKind::Cuda || maximum_tokens == 0 {
            return Ok(maximum_tokens);
        }
        if portable_state_copies == 0 {
            return Err(Error::InvalidInput(
                "portable managed-state copy count must be positive".into(),
            ));
        }
        let Some(authority) = self.resource_authority.as_ref() else {
            return Ok(maximum_tokens.min(4_096));
        };
        let planning = authority.planning_headroom_bytes(self.worker_backend)?;
        let (growth_headroom, safety_reserve_bytes) = match headroom {
            PortableFitHeadroom::Planning { .. } => (planning, safety_reserve_bytes),
            // Retained state keeps the one reserve its invocation fit keeps
            // too: the configured floor or, for an observed load, the
            // measured untracked growth (ADM-11).
            PortableFitHeadroom::Live { reserve, .. } => (
                authority.state_fit_headroom_bytes(self.worker_backend)?,
                match reserve {
                    StateFitReserve::MeasuredUntracked => authority
                        .state_safety_reserve_bytes(self.worker_backend, safety_reserve_bytes),
                    StateFitReserve::Configured => safety_reserve_bytes,
                },
            ),
        };
        let ResourceAmount::Known(headroom_bytes) = planning else {
            return Ok(maximum_tokens.min(4_096));
        };
        // The model-authored minimum is admitted against planning headroom;
        // live memory caps only growth above it. Right after a large load the
        // live sample undercounts what the OS reclaims on demand (clean
        // file-backed pages of the checkpoint just read, compressible
        // pages), so a live floor refused even a one-page state.
        let floor_budget_bytes = headroom_bytes.saturating_sub(safety_reserve_bytes);
        let growth_budget_bytes = match growth_headroom {
            ResourceAmount::Known(bytes) => bytes.saturating_sub(safety_reserve_bytes),
            ResourceAmount::Unknown => floor_budget_bytes,
        }
        .min(floor_budget_bytes);
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".into()))?;
        let state_plan = negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: self.worker_backend,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
            },
        )?;
        let prefix_snapshot_rows = self.prefix_snapshot_rows_for(contract);
        let required_bytes = |tokens: u64| -> Result<u64> {
            let (allocation, _) = plan_managed_state_capacity_with_policy(
                &state_plan,
                model_instance,
                ManagedStateCapacityRequest {
                    total_paged_pages: u32::MAX,
                    logical_token_reach: Some(tokens),
                    retained_sequence_rows,
                    staged_transaction_rows,
                },
                false,
                prefix_snapshot_rows,
            )?;
            let resources = managed_state_resources(
                self.worker_backend,
                allocation.maximum_resources(&state_plan)?,
            )?;
            let bytes = match self.worker_backend {
                BackendKind::Cpu => known_resource_bytes(resources.host_bytes, "host"),
                BackendKind::Metal => known_resource_bytes(resources.unified_bytes, "unified"),
                BackendKind::Cuda => unreachable!("CUDA bypassed portable fitting"),
            }?;
            bytes
                .checked_mul(u64::from(portable_state_copies))
                .ok_or_else(|| {
                    Error::ModelLoadError("aggregate managed-state byte plan overflow".into())
                })
        };

        let floor_tokens = match headroom {
            PortableFitHeadroom::Live { minimum_tokens, .. }
            | PortableFitHeadroom::Planning { minimum_tokens } => minimum_tokens,
        };
        let minimum_tokens = u64::from(page_tokens_hint)
            .max(floor_tokens)
            .min(maximum_tokens);
        let minimum_bytes = required_bytes(minimum_tokens)?;
        if minimum_bytes > floor_budget_bytes {
            tracing::warn!(
                "portable context cannot fit the model-authored minimum: minimum_tokens={minimum_tokens}, state_bytes={minimum_bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}, shortfall_bytes={}",
                minimum_bytes - floor_budget_bytes
            );
            let message = crate::engine::resources::capacity_shortfall_reason(
                format_args!(
                    "not enough memory for this model's minimum {minimum_tokens}-token state"
                ),
                minimum_bytes,
                floor_budget_bytes,
            );
            // A model-authored floor (ASR, TTS, chat) that headroom cannot
            // hold is a capacity refusal (503), not a load defect.
            return Err(match headroom {
                PortableFitHeadroom::Planning { minimum_tokens: 0 } => {
                    Error::ModelLoadError(message)
                }
                _ => Error::Overloaded(message),
            });
        }
        let budget_bytes = growth_budget_bytes.max(minimum_bytes);
        if required_bytes(maximum_tokens)? <= budget_bytes {
            return Ok(maximum_tokens);
        }

        let (mut low, mut high) = (minimum_tokens, maximum_tokens);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if required_bytes(middle)? <= budget_bytes {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(low)
    }

    /// Fit a CUDA logical reach whose complete paged backing is materialized
    /// before Ready publication. This turns the live-capacity observation into
    /// physical ownership instead of a revocable promise to grow later.
    ///
    /// Explicit rows are a hard contract. Automatic retained rows shrink to
    /// the largest count whose recurrent state fits the configured share of
    /// the budget, never below the staged (decode) width; the rest of the
    /// budget backs KV pages.
    pub(crate) fn fit_cuda_resident_logical_token_reach(
        &self,
        model_instance: ModelInstanceId,
        contract: &InferenceStateContract,
        maximum_tokens_for_rows: impl Fn(u32) -> Result<u64>,
        safety_reserve_bytes: u64,
        page_tokens_hint: usize,
        rows: CudaResidentRowRequest,
    ) -> Result<CudaResidentFit> {
        let requested = |logical_token_reach| CudaResidentFit {
            logical_token_reach,
            retained_sequence_rows: rows.retained_sequence_rows,
            staged_transaction_rows: rows.staged_transaction_rows,
            prefix_snapshot_rows: 0,
            per_sequence_state_bytes: 0,
            resident_state_bytes: 0,
        };
        let maximum_tokens = maximum_tokens_for_rows(rows.retained_sequence_rows)?;
        if self.worker_backend != BackendKind::Cuda || maximum_tokens == 0 {
            return Ok(requested(maximum_tokens));
        }
        let Some(authority) = self.resource_authority.as_ref() else {
            return Err(Error::ModelLoadError(
                "CUDA managed context fitting requires a resource authority".into(),
            ));
        };
        let ResourceAmount::Known(headroom_bytes) =
            authority.planning_headroom_bytes(BackendKind::Cuda)?
        else {
            return Err(Error::ModelLoadError(
                "CUDA managed context fitting requires known device capacity".into(),
            ));
        };
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".into()))?;
        let state_plan = negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cuda,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
            },
        )?;
        fit_resident_rows_and_reach(
            &state_plan,
            model_instance,
            maximum_tokens_for_rows,
            headroom_bytes,
            safety_reserve_bytes,
            page_tokens_hint,
            rows,
            self.prefix_snapshot_rows_for(contract),
        )
    }

    /// Device memory the worker's resource authority reports in total.
    pub(crate) fn device_capacity(&self) -> ResourceAmount {
        self.resource_authority
            .as_ref()
            .map_or(ResourceAmount::Unknown, |authority| {
                authority.snapshot().physical.capacity.device_bytes
            })
    }

    #[cfg(test)]
    pub(crate) fn with_prefix_cache_salt(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
    ) -> Self {
        Self::with_prefix_cache_policy(resource_authority, salt, usize::MAX)
    }

    pub(crate) fn with_prefix_cache_policy(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
        max_prefix_cache_pages: usize,
    ) -> Self {
        let mut manager = Self::new(resource_authority);
        manager.prefix_cache_salt = salt;
        manager.max_prefix_cache_pages = max_prefix_cache_pages;
        manager
    }

    pub(crate) fn for_worker_with_prefix_cache_policy(
        resource_authority: Option<Arc<ResourceAuthority>>,
        salt: Option<[u8; 32]>,
        max_prefix_cache_pages: usize,
        backend: BackendKind,
        device: Device,
    ) -> Self {
        let mut manager = Self::for_worker(resource_authority, backend, device);
        manager.prefix_cache_salt = salt;
        manager.max_prefix_cache_pages = max_prefix_cache_pages;
        manager
    }

    /// Enables DS4 hierarchical offload with the resolved policy. A `None`
    /// policy (no host budget) keeps offload fully dormant.
    pub(crate) fn set_host_offload_policy(&mut self, policy: Option<HostOffloadPolicy>) {
        self.host_offload_policy = policy;
    }

    /// Committed prefix snapshots each snapshot-sharing model may hold (its
    /// snapshot budget, in rows of hybrid state). Takes effect for models
    /// loaded afterwards.
    pub(crate) fn set_prefix_snapshot_rows(&mut self, rows: u32) {
        self.prefix_snapshot_rows = rows;
    }

    /// Whether the prefix page budget was left unset (see
    /// [`prefix_index_pages`]). Takes effect for models loaded afterwards.
    pub(crate) fn set_automatic_prefix_cache_pages(&mut self, automatic: bool) {
        self.automatic_prefix_cache_pages = automatic;
    }

    /// The snapshot rows a model with `contract` is fitted and reserved for:
    /// the configured budget when its hybrid contract shares committed
    /// snapshots and prefix caching is engaged (a namespace and a page
    /// budget), else zero. A snapshot outlives its publisher's committed
    /// state, so these bytes are authorized on the model lease at load,
    /// recorded as materialized while the index pins them, and released when
    /// it evicts them; they never grow memory outside the reservation.
    pub(crate) fn prefix_snapshot_rows_for(&self, contract: &InferenceStateContract) -> u32 {
        if self.prefix_cache_salt.is_some()
            && self.max_prefix_cache_pages > 0
            && tensor_snapshot_prefix_policy(contract).is_some()
        {
            self.prefix_snapshot_rows
        } else {
            0
        }
    }

    #[cfg(test)]
    pub(crate) fn telemetry_snapshot(&self) -> ManagedKvTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    pub(crate) fn runtime_snapshot(&self) -> ManagedKvRuntimeSnapshot {
        let mut totals = ManagedKvRuntimeTotalsSnapshot {
            models: usize_to_u64(self.models.len()),
            ..ManagedKvRuntimeTotalsSnapshot::default()
        };
        let mut models = self
            .models
            .iter()
            .map(|(model_instance, state)| {
                totals.registered_sessions = totals
                    .registered_sessions
                    .saturating_add(usize_to_u64(state.registered_sessions.len()));
                totals.physical_bytes = totals
                    .physical_bytes
                    .saturating_add(state.runtime.physical_bytes());
                totals.resident_paged_bytes = totals
                    .resident_paged_bytes
                    .saturating_add(state.runtime.resident_paged_bytes());
                totals.authorized_tensor_bytes = totals
                    .authorized_tensor_bytes
                    .saturating_add(state.runtime.authorized_tensor_bytes());
                totals.materialized_tensor_bytes = totals
                    .materialized_tensor_bytes
                    .saturating_add(state.tensor_materialized_bytes);
                let mut arenas = state
                    .runtime
                    .plan
                    .groups
                    .iter()
                    .map(|group| {
                        let coordinator = state
                            .coordinators
                            .get(&group.arena)
                            .expect("resolved managed arena has a coordinator")
                            .stats();
                        let admission_claimed_pages = state
                            .capacity_claims
                            .values()
                            .flat_map(|claims| claims.iter())
                            .filter_map(|(arena, pages)| {
                                (*arena == group.arena).then_some(u64::from(*pages))
                            })
                            .sum::<u64>();
                        let admission_claims = state
                            .capacity_claims
                            .values()
                            .filter(|claims| claims.iter().any(|(arena, _)| *arena == group.arena))
                            .count();
                        let coordinator = ManagedKvCoordinatorSnapshot {
                            capacity_pages: usize_to_u64(coordinator.capacity_pages),
                            allocated_pages: usize_to_u64(coordinator.allocated_pages),
                            free_pages: usize_to_u64(coordinator.free_pages),
                            admission_claimed_pages,
                            admission_available_pages: usize_to_u64(coordinator.capacity_pages)
                                .saturating_sub(admission_claimed_pages),
                            admission_claims: usize_to_u64(admission_claims),
                            table_refs: usize_to_u64(coordinator.table_refs),
                            prefix_refs: usize_to_u64(coordinator.prefix_refs),
                            execution_pins: usize_to_u64(coordinator.execution_pins),
                            transfer_pins: usize_to_u64(coordinator.transfer_pins),
                            reservations: usize_to_u64(coordinator.reservations),
                            active_transactions: usize_to_u64(coordinator.active_transactions),
                        };
                        let arena = state
                            .runtime
                            .arenas
                            .get(&group.arena)
                            .expect("resolved managed arena has physical storage");
                        let operation_stats = arena.operation_stats();
                        let operations = ManagedKvOperationSnapshot {
                            slot_write_dispatches: operation_stats.slot_write_dispatches,
                            paged_prefill_dispatches: operation_stats.paged_prefill_dispatches,
                            paged_decode_dispatches: operation_stats.paged_decode_dispatches,
                            page_zero_dispatches: operation_stats.page_zero_dispatches,
                            page_copy_dispatches: operation_stats.page_copy_dispatches,
                            backing_allocations: operation_stats.backing_allocations.unwrap_or(0),
                            backing_allocations_observed_arenas: u64::from(
                                operation_stats.backing_allocations.is_some(),
                            ),
                            workspace_bytes: operation_stats.workspace_bytes.unwrap_or(0),
                            workspace_bytes_observed_arenas: u64::from(
                                operation_stats.workspace_bytes.is_some(),
                            ),
                            workspace_budget_bytes: operation_stats
                                .workspace_budget_bytes
                                .unwrap_or(0),
                            workspace_budget_bytes_observed_arenas: u64::from(
                                operation_stats.workspace_budget_bytes.is_some(),
                            ),
                            workspace_high_water_bytes: operation_stats
                                .workspace_high_water_bytes
                                .unwrap_or(0),
                            workspace_high_water_bytes_observed_arenas: u64::from(
                                operation_stats.workspace_high_water_bytes.is_some(),
                            ),
                            workspace_allocations: operation_stats
                                .workspace_allocations
                                .unwrap_or(0),
                            workspace_allocations_observed_arenas: u64::from(
                                operation_stats.workspace_allocations.is_some(),
                            ),
                            host_synchronizations: operation_stats.host_synchronizations,
                            cpu_reference_attention_dispatches: operation_stats
                                .cpu_reference_attention_dispatches,
                            portable_attention_dispatches: operation_stats
                                .portable_attention_dispatches,
                            cuda_native_attention_dispatches: operation_stats
                                .cuda_native_attention_dispatches,
                            cuda_flash_attention_dispatches: operation_stats
                                .cuda_flash_attention_dispatches,
                            cuda_split_decode_dispatches: operation_stats
                                .cuda_split_decode_dispatches,
                            metal_native_attention_dispatches: operation_stats
                                .metal_native_attention_dispatches,
                            cuda_graph_warmups: operation_stats.cuda_graph_warmups,
                            cuda_graph_captures: operation_stats.cuda_graph_captures,
                            cuda_graph_replays: operation_stats.cuda_graph_replays,
                            cuda_graph_fallbacks: operation_stats.cuda_graph_fallbacks,
                            cuda_graph_backoff_hits: operation_stats.cuda_graph_backoff_hits,
                            cuda_graph_evictions: operation_stats.cuda_graph_evictions,
                        };
                        add_coordinator_stats(&mut totals.coordinator, &coordinator);
                        totals.operations.add_assign(operations.clone());
                        totals.arenas = totals.arenas.saturating_add(1);
                        ManagedKvArenaRuntimeSnapshot {
                            generation: group.arena.generation,
                            group_id: group.id.get(),
                            domain_id: group.domain.get(),
                            device_ordinal: group.arena.device_ordinal,
                            page_tokens: group.page_tokens,
                            storage_dtype: format!("{:?}", group.storage.dtype())
                                .to_ascii_lowercase(),
                            token_capacity: u64::from(group.capacity_pages)
                                .saturating_mul(u64::from(group.page_tokens)),
                            bytes_per_page: group.bytes_per_page,
                            physical_bytes: arena.resident_bytes(),
                            prefix_capacity_pages: state
                                .prefix_indexes
                                .get(&group.arena)
                                .map_or(0, |index| usize_to_u64(index.capacity_pages())),
                            coordinator,
                            operations,
                        }
                    })
                    .collect::<Vec<_>>();
                arenas.sort_by_key(|arena| (arena.generation, arena.group_id));
                let tensor_capacity = state.runtime.tensor_state().map(|arena| arena.capacity());
                ManagedKvModelRuntimeSnapshot {
                    model_instance: *model_instance,
                    plan_fingerprint: state.runtime.plan.fingerprint().to_string(),
                    state_plan_v2_fingerprint: state
                        .runtime
                        .state_plan_v2
                        .fingerprint()
                        .to_string(),
                    backend: state.runtime.plan.backend,
                    device_ordinal: state.runtime.plan.device_ordinal,
                    resident_paged_bytes: state.runtime.resident_paged_bytes(),
                    authorized_tensor_bytes: state.runtime.authorized_tensor_bytes(),
                    materialized_tensor_bytes: state.tensor_materialized_bytes,
                    physical_bytes: state.runtime.physical_bytes(),
                    registered_sessions: usize_to_u64(state.registered_sessions.len()),
                    single_sequence_token_capacity: state.runtime.maximum_sequence_tokens(),
                    aggregate_token_capacity: state.runtime.logical_token_reach(),
                    full_context_sequence_capacity: full_context_sequence_capacity(
                        &state.runtime.plan,
                        state.runtime.maximum_sequence_tokens(),
                    ),
                    retained_sequence_capacity: tensor_capacity
                        .map(|capacity| capacity.sequence_capacity()),
                    staged_transaction_capacity: tensor_capacity
                        .map(|capacity| capacity.transaction_capacity()),
                    tensor_bytes_per_sequence: tensor_capacity
                        .map(|capacity| capacity.per_sequence_bytes()),
                    prefix_snapshot_capacity: tensor_capacity
                        .map(|capacity| capacity.snapshot_capacity()),
                    prefix_snapshot_budget_bytes: tensor_capacity
                        .map_or(0, |capacity| capacity.snapshot_capacity_bytes()),
                    prefix_snapshots: state
                        .tensor_snapshots
                        .as_ref()
                        .map_or(0, |sharing| usize_to_u64(sharing.index.len())),
                    prefix_snapshot_bytes: state
                        .tensor_snapshots
                        .as_ref()
                        .map_or(0, |sharing| sharing.index.total_bytes()),
                    incremental_claim_sessions: usize_to_u64(
                        state.incremental_claim_sessions.len(),
                    ),
                    arenas,
                }
            })
            .collect::<Vec<_>>();
        models.sort_by_key(|model| model.model_instance);
        let mut counters = self.telemetry.snapshot();
        counters.prefix_retained_pages = totals.coordinator.prefix_refs;
        counters.kv_host_pages = self.models.values().fold(0_u64, |total, state| {
            total
                + state
                    .host_offload
                    .as_ref()
                    .map(|offload| {
                        offload
                            .arenas
                            .values()
                            .map(|arena| arena.pool.resident_pages() as u64)
                            .sum::<u64>()
                    })
                    .unwrap_or(0)
        });
        ManagedKvRuntimeSnapshot {
            memory_accounting: "resident_paged_plus_authorized_tensor",
            totals,
            counters,
            models,
        }
    }

    pub(crate) fn worker_backend(&self) -> BackendKind {
        self.worker_backend
    }

    pub(crate) fn capacity_claim_sessions(
        &self,
        model_instance: ModelInstanceId,
    ) -> Vec<SessionKey> {
        self.models
            .get(&model_instance)
            .map(|state| state.capacity_claims.keys().cloned().collect())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn bind_request(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity_pages: usize,
        page_tokens_hint: usize,
        capability: &InferenceStateCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let capacity = ManagedStateCapacityRequest {
            total_paged_pages: u32::try_from(capacity_pages).map_err(|_| {
                Error::InvalidInput("managed KV page capacity exceeds u32".to_string())
            })?,
            logical_token_reach: None,
            retained_sequence_rows: u32::try_from(capacity_pages).map_err(|_| {
                Error::InvalidInput("managed KV sequence capacity exceeds u32".to_string())
            })?,
            staged_transaction_rows: u32::try_from(capacity_pages).map_err(|_| {
                Error::InvalidInput("managed KV transaction capacity exceeds u32".to_string())
            })?,
        };
        self.bind_model_state_with_capacity(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            contract,
        )
        .map(Some)
    }

    #[cfg(test)]
    pub(crate) fn bind_request_with_capacity(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        capability: &InferenceStateCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        self.bind_request_with_capacity_policy(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            capability,
            false,
        )
    }

    pub(crate) fn bind_request_with_capacity_policy(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        capability: &InferenceStateCapability,
        materialize_cuda_paged_at_load: bool,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        self.bind_model_state_with_capacity_policy(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            contract,
            materialize_cuda_paged_at_load,
        )
        .map(Some)
    }

    pub(crate) fn bind_model_state_with_capacity(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        contract: &InferenceStateContract,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        self.bind_model_state_with_capacity_policy(
            model_instance,
            backend,
            capacity,
            page_tokens_hint,
            contract,
            false,
        )
    }

    fn bind_model_state_with_capacity_policy(
        &mut self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capacity: ManagedStateCapacityRequest,
        page_tokens_hint: usize,
        contract: &InferenceStateContract,
        materialize_cuda_paged_at_load: bool,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        validate_sliding_contract(contract, backend)?;
        if let Some(state) = self.models.get(&model_instance) {
            if &state.contract != contract
                || state.runtime.plan.backend != backend
                || state.runtime.state_plan_v2.contract_fingerprint != contract.fingerprint()?
            {
                return Err(Error::InvalidInput(
                    "one loaded model instance published incompatible managed KV contracts"
                        .to_string(),
                ));
            }
            return Ok(state.runtime.clone());
        }
        if backend != self.worker_backend {
            return Err(Error::InvalidInput(format!(
                "managed KV request targets {backend:?}, but its worker is bound to {:?}",
                self.worker_backend
            )));
        }
        let backend_runtime = self.backend_runtime.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                self.backend_unavailable
                    .clone()
                    .unwrap_or_else(|| format!("managed KV is unavailable for {backend:?}")),
            )
        })?;
        let page_tokens_hint = u32::try_from(page_tokens_hint)
            .map_err(|_| Error::InvalidInput("managed KV page size exceeds u32".to_string()))?;
        let first_arena_generation = self.next_arena_generation;
        let state_plan_v2 = negotiate_state_plan(
            contract,
            &StateBackendPlanRequest {
                backend,
                device_ordinal: self.worker_device_ordinal,
                page_tokens_hint: Some(page_tokens_hint),
                storage_dtype_hint: None,
            },
        )?;
        let (allocation_plan, tensor_capacity) = plan_managed_state_capacity_with_policy(
            &state_plan_v2,
            model_instance,
            capacity,
            materialize_cuda_paged_at_load,
            self.prefix_snapshot_rows_for(contract),
        )?;
        let plan = ResolvedKvPlan::from_runtime_allocation(
            first_arena_generation,
            &state_plan_v2,
            &allocation_plan,
        )?;
        let tensor_state = tensor_capacity
            .map(|capacity| {
                TensorStateArena::new_with_contract(
                    Arc::new(state_plan_v2.clone()),
                    contract,
                    capacity,
                    self.worker_device.clone(),
                )
            })
            .transpose()?
            .map(Arc::new);

        let maximum_state_resources = allocation_plan.maximum_resources(&state_plan_v2)?;
        let resources = managed_state_resources(backend, maximum_state_resources)?;
        let mut materialized_resources = managed_state_resources(
            backend,
            allocation_plan.initial_state_resources(&state_plan_v2)?,
        )?;
        // A fitted CUDA context is a durable allocation promise, not a
        // one-time observation. Reserve the worst contiguous-replacement peak
        // before Ready publication so later request/workspace allocations
        // cannot consume memory required by lazy KV growth.
        //
        // DS4: the host pool's byte ceiling is part of the model's durable
        // envelope, so the lease authorizes it up front (same charge shape as
        // the materialization below) and the load-time charge cannot exceed
        // the reservation. The supervisor has already validated the budget
        // against the assignment's host/shared limit.
        let mut authorization = if backend == BackendKind::Cuda {
            cuda_managed_state_peak_authorization(resources, &plan.groups)?
        } else {
            resources
        };
        if let Some(policy) = self
            .host_offload_policy
            .filter(|_| self.max_prefix_cache_pages > 0)
        {
            let charge = match backend {
                BackendKind::Metal => ResourceVector {
                    unified_bytes: ResourceAmount::Known(policy.budget_bytes),
                    ..ResourceVector::zero()
                },
                _ => ResourceVector {
                    host_bytes: ResourceAmount::Known(policy.budget_bytes),
                    ..ResourceVector::zero()
                },
            };
            authorization = authorization.checked_add(charge)?;
        }
        let resource_lease = self
            .resource_authority
            .as_ref()
            .map(|authority| {
                reserve_managed_arena(authority, model_instance, backend, authorization)
            })
            .transpose()?;
        let mut arenas = HashMap::with_capacity(plan.groups.len());
        let mut coordinators = HashMap::with_capacity(plan.groups.len());
        let mut prefix_indexes = HashMap::with_capacity(plan.groups.len());
        let mut allocation_ledger = StateAllocationLedger::new(&allocation_plan);
        for group in &plan.groups {
            let config = arena_config(contract, group)?;
            let arena = backend_runtime.allocate_arena(config)?;
            if arena.backend_kind() != self.worker_backend {
                return Err(Error::InferenceError(format!(
                    "managed KV runtime allocated a {:?} arena for a {:?} worker",
                    arena.backend_kind(),
                    self.worker_backend
                )));
            }
            if arena.device_location() != self.worker_device_location {
                return Err(Error::InferenceError(format!(
                    "managed KV runtime allocated arena device {:?} for exact worker device {:?}",
                    arena.device_location(),
                    self.worker_device_location
                )));
            }
            let resident_pages = arena.resident_capacity_pages();
            if arenas.insert(group.arena, arena).is_some() {
                return Err(Error::InferenceError(
                    "resolved KV plan reused one arena identity".to_string(),
                ));
            }
            let requested_owned_bytes = group
                .bytes_per_page
                .checked_mul(u64::from(resident_pages))
                .ok_or_else(|| Error::Overloaded("managed KV byte total overflow".into()))?;
            allocation_ledger.reconcile_group_receipt(
                &allocation_plan,
                crate::kv::v2::StateGroupId::new(group.id.get()),
                group.domain,
                AllocationReceipt {
                    requested_owned_bytes,
                    committed_owned_bytes: requested_owned_bytes,
                    allocator_overhead_bytes: 0,
                    residency: ResidencyMeasurement::Unknown,
                },
            )?;
            self.telemetry.record_backing_allocation();
            coordinators.insert(
                group.arena,
                KvCacheCoordinator::new(group.arena, group.capacity_pages as usize),
            );
            prefix_indexes.insert(
                group.arena,
                CoordinatedPrefixIndex::with_telemetry(
                    prefix_index_pages(
                        self.max_prefix_cache_pages,
                        self.automatic_prefix_cache_pages,
                        backend,
                        materialize_cuda_paged_at_load,
                        group.capacity_pages,
                    ),
                    self.telemetry.clone(),
                ),
            );
        }
        allocation_ledger.ensure_ready(&allocation_plan)?;
        if let Some(lease) = resource_lease.as_ref() {
            // CUDA paged arenas materialize only their sealed initial growth
            // quantum. CPU/Metal remain fully backed; tensor capacity remains
            // demand allocated.
            lease.record_materialized_usage(materialized_resources)?;
        }
        // Tensor-state capacity is independently authorized and may materialize
        // lazily. Keep that sealed envelope in model accounting while paged
        // CUDA backing reports only its current resident extent.
        let non_paged_physical_bytes = tensor_state
            .as_ref()
            .map(|arena| arena.capacity().authorized_bytes())
            .unwrap_or(0);
        // DS4: build the host tier when offload is engaged and committed
        // prefix pages exist to demote. The pool's byte ceiling is charged to
        // the node memory ledger up front (DINV-05) — the budget is the
        // operator's declared ceiling for host-tier pages.
        let host_offload = match self
            .host_offload_policy
            .filter(|_| self.max_prefix_cache_pages > 0)
        {
            Some(policy) => {
                let mut offload = offload::ModelOffload {
                    arenas: HashMap::with_capacity(plan.groups.len()),
                };
                // The model lease is charged one budget, so the arenas share
                // it: giving each arena the whole budget let a multi-domain
                // model keep `arenas x budget` bytes resident on the host.
                let per_arena_budget = split_host_offload_budget(policy.budget_bytes, arenas.len());
                for arena in arenas.values() {
                    offload.arenas.insert(
                        arena.id(),
                        offload::ArenaOffload::new(
                            crate::backends::kv::arena_page_bytes(arena.config()),
                            per_arena_budget,
                        )?,
                    );
                }
                let charge = match backend {
                    BackendKind::Metal => ResourceVector {
                        unified_bytes: ResourceAmount::Known(policy.budget_bytes),
                        ..ResourceVector::zero()
                    },
                    _ => ResourceVector {
                        host_bytes: ResourceAmount::Known(policy.budget_bytes),
                        ..ResourceVector::zero()
                    },
                };
                materialized_resources = materialized_resources.checked_add(charge)?;
                if let Some(lease) = resource_lease.as_ref() {
                    lease.record_materialized_usage(materialized_resources)?;
                }
                Some(offload)
            }
            None => None,
        };
        // DS1.2b: derive the snapshot-sharing surface from the strict contract
        // gate. The declared snapshot interval must land on the resolved page
        // grid. The index is bounded by the snapshot budget the tensor arena
        // authorized beside its rows (zero rows without a prefix namespace).
        let tensor_snapshots = tensor_snapshot_prefix_policy(contract).and_then(|policy| {
            let group = plan
                .groups
                .iter()
                .find(|group| group.domain == policy.paged_domain)?;
            let capacity = tensor_capacity.as_ref()?;
            let interval_tokens = u32::try_from(policy.interval_steps).ok()?;
            if interval_tokens == 0
                || !policy
                    .interval_steps
                    .is_multiple_of(u64::from(group.page_tokens))
            {
                // Hybrid prefix reuse is unusable without snapshots, so say
                // why instead of silently serving every turn cold.
                tracing::warn!(
                    ?model_instance,
                    snapshot_interval_tokens = policy.interval_steps,
                    page_tokens = group.page_tokens,
                    "Recurrent-state snapshot interval is not a multiple of the KV page size; hybrid prefix reuse is disabled for this model"
                );
                return None;
            }
            Some(TensorSnapshotSharing {
                paged_arena: group.arena,
                interval_tokens,
                index: TensorStateSnapshotIndex::with_telemetry(
                    usize::try_from(capacity.snapshot_capacity()).unwrap_or(0),
                    capacity.snapshot_capacity_bytes(),
                    self.telemetry.clone(),
                ),
                policy,
            })
        });
        let runtime = Arc::new(ManagedKvModelRuntime {
            plan: Arc::new(plan),
            state_plan_v2: Arc::new(state_plan_v2),
            allocation_plan: Arc::new(allocation_plan),
            arenas,
            tensor_state,
            non_paged_physical_bytes,
            maximum_sequence_tokens: AtomicU64::new(0),
        });
        self.models.insert(
            model_instance,
            ManagedKvModelState {
                closing: false,
                contract: contract.clone(),
                runtime: runtime.clone(),
                coordinators,
                prefix_indexes,
                pending_prefixes: HashMap::new(),
                pending_promotions: HashMap::new(),
                exact_target_transactions: HashSet::new(),
                registered_sessions: HashSet::new(),
                session_generations: HashMap::new(),
                capacity_claims: HashMap::new(),
                incremental_claim_sessions: HashSet::new(),
                tensor_sequences: HashMap::new(),
                tensor_snapshots,
                host_offload,
                resource_lease,
                materialized_resources,
                tensor_materialized_bytes: 0,
                tensor_rows_request_priced: false,
                allocation_ledger,
            },
        );
        self.next_arena_generation = first_arena_generation
            .checked_add(u32::try_from(runtime.plan.groups.len()).map_err(|_| {
                Error::InvalidInput("managed KV arena count exceeds u32".to_string())
            })?)
            .ok_or_else(|| Error::InvalidInput("managed KV arena generation overflow".into()))?;
        Ok(runtime)
    }

    /// Stop charging a model's lazy tensor rows as pending live claims, for
    /// a model whose requests reserve the rows they create (Qwen3-TTS prices
    /// its encoder memory in the prefill workspace, then commits it). The
    /// envelope stays authorized (planning still holds it); recording it as
    /// materialized means a row is counted once, by its request, and the
    /// rows no request has created never refuse a request whose own need
    /// fits live memory. Idempotent.
    pub(crate) fn price_tensor_rows_with_requests(
        &mut self,
        model_instance: ModelInstanceId,
    ) -> Result<()> {
        let state = self.models.get_mut(&model_instance).ok_or_else(|| {
            Error::InvalidInput("managed tensor rows target an unbound model".into())
        })?;
        if state.tensor_rows_request_priced {
            return Ok(());
        }
        let Some(arena) = state.runtime.tensor_state() else {
            return Err(Error::InvalidInput(
                "request-priced tensor rows need a model with tensor state".into(),
            ));
        };
        let envelope = arena.capacity().authorized_bytes();
        if let Some(lease) = state.resource_lease.as_ref() {
            lease.record_materialized_usage(state.materialized_resources.checked_add(
                tensor_state_resources(state.runtime.plan.backend, envelope)?,
            )?)?;
        }
        state.tensor_materialized_bytes = envelope;
        state.tensor_rows_request_priced = true;
        Ok(())
    }

    /// Resolve an arena runtime that was allocated by model loading. Request
    /// admission must never create backing storage or expand model residency.
    #[cfg(test)]
    pub(crate) fn require_loaded_runtime(
        &self,
        model_instance: ModelInstanceId,
        backend: BackendKind,
        capability: &InferenceStateCapability,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let state = self.models.get(&model_instance).ok_or_else(|| {
            Error::InferenceError(
                "loaded adapter published managed KV without load-time physical allocation"
                    .to_string(),
            )
        })?;
        if state.closing {
            return Err(Error::InferenceError(
                "managed KV runtime is closing".to_string(),
            ));
        }
        if backend != self.worker_backend
            || state.runtime.plan.backend != backend
            || &state.contract != contract
        {
            return Err(Error::InferenceError(
                "load-time managed KV runtime does not match the request capability".to_string(),
            ));
        }
        Ok(Some(state.runtime.clone()))
    }

    #[cfg(test)]
    pub(crate) fn prepare(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
    ) -> Result<Option<ManagedCacheReservation>> {
        self.prepare_with_admission(
            runtime,
            txn_id,
            session,
            work,
            request,
            ManagedClaimPolicy::Full,
            true,
        )
    }

    /// Prepare one scheduled row. `prefix_attach` is false when the row's
    /// adapter cannot resume prefill above a non-zero cursor: a full-prefill
    /// row always runs its whole prompt from position 0, so attaching shared
    /// prefix pages would hand it a cache that does not start empty.
    pub(crate) fn prepare_row(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
        claim: ManagedClaimPolicy,
        prefix_attach: bool,
    ) -> Result<Option<ManagedCacheReservation>> {
        self.prepare_with_admission(
            runtime,
            txn_id,
            session,
            work,
            request,
            claim,
            prefix_attach,
        )
    }

    /// DS1.5: read-only admission probe for the managed prefix cursor.
    ///
    /// Returns the largest committed boundary the first prefill span may
    /// start above: complete shared pages plus, for snapshot-sharing
    /// contracts, a committed tensor snapshot at that exact boundary. The
    /// scheduler plans the first span from this cursor; prepare re-verifies
    /// the same lookup transactionally before any attach is staged, so a
    /// boundary that disappears between admission and execution degrades to
    /// the cursor-lost retry instead of a stale attach.
    pub(crate) fn probe_managed_prefix(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        request: &EngineCoreRequest,
    ) -> Result<Option<u32>> {
        let namespace = managed_prefix_namespace(Some(request), runtime, self.prefix_cache_salt)?;
        let Some(namespace) = namespace else {
            return Ok(None);
        };
        let Some(state) = self.models.get_mut(&runtime.plan.model_instance) else {
            return Ok(None);
        };
        if state.closing {
            return Ok(None);
        }
        // The cursor must leave at least one private prompt token; a fully
        // cached prompt has no executable first span.
        let prompt_tokens = request.prompt_tokens.as_slice();
        if prompt_tokens.len() < 2 {
            return Ok(None);
        }
        let reusable = prompt_tokens.len() - 1;
        let max_host_pages = self
            .host_offload_policy
            .map(|policy| policy.max_promotion_pages)
            .unwrap_or(0);
        for group in runtime.plan.groups.iter() {
            if !prefix_enabled_for_domain(&state.contract, group.domain) {
                continue;
            }
            let chain = state.host_offload.as_mut().and_then(|offload| {
                offload
                    .arenas
                    .get_mut(&group.arena)
                    .map(|arena| &mut arena.chain)
            });
            let prefix_index = state.prefix_indexes.get_mut(&group.arena).ok_or_else(|| {
                Error::InferenceError(
                    "resolved arena has no prefix index for the prefix probe".into(),
                )
            })?;
            let mut matched = match chain {
                Some(chain) => offload::lookup_longest_with_host(
                    &mut *prefix_index,
                    chain,
                    &namespace,
                    &prompt_tokens[..reusable],
                    group.page_tokens,
                    max_host_pages,
                )
                .map_err(prefix_error)?,
                None => prefix_index
                    .lookup_longest(&namespace, &prompt_tokens[..reusable], group.page_tokens)
                    .map_err(prefix_error)?,
            };
            if matched.reused_tokens == 0 {
                continue;
            }
            // Snapshot-sharing contracts may only start above a boundary that
            // also carries a committed tensor snapshot; walk the matched chain
            // back the same way prepare's reconcile does, without touching
            // attach/truncation telemetry (no fork happens here).
            if let Some(sharing) = state.tensor_snapshots.as_mut() {
                if sharing.paged_arena == group.arena && sharing.policy.paged_domain == group.domain
                {
                    while !matched.page_digests.is_empty() {
                        let boundary = u64::from(matched.reused_tokens);
                        let digest = matched
                            .page_digests
                            .last()
                            .copied()
                            .expect("matched chain holds complete pages");
                        if sharing.index.lookup(digest, boundary)?.is_some() {
                            return Ok(Some(matched.reused_tokens));
                        }
                        matched.pop_last_page(group.page_tokens);
                    }
                    continue;
                }
            }
            return Ok(Some(matched.reused_tokens));
        }
        Ok(None)
    }

    /// SCH-F7: [`Self::probe_managed_prefix`] for one exact session at
    /// schedule time. It offers a cursor only where prepare would attach it:
    /// an initial session generation (a reset generation must rebuild from
    /// context 0), no committed table tokens, and no live tensor sequence that
    /// a snapshot fork would have to cross. Anything else would lose the
    /// cursor at prepare and replan the request.
    pub(crate) fn probe_managed_prefix_for_session(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        request: &EngineCoreRequest,
        session: &SessionKey,
    ) -> Result<Option<u32>> {
        if let Some(state) = self.models.get(&runtime.plan.model_instance) {
            let restarted = state
                .session_generations
                .get(session)
                .is_some_and(|generation| *generation != ManagedSessionGeneration::INITIAL);
            let committed = state.runtime.plan.groups.iter().any(|group| {
                state
                    .coordinators
                    .get(&group.arena)
                    .and_then(|coordinator| coordinator.snapshot(session, group.domain).ok())
                    .is_some_and(|snapshot| snapshot.committed_tokens > 0)
            });
            if restarted || committed || state.tensor_sequences.contains_key(session) {
                return Ok(None);
            }
        }
        self.probe_managed_prefix(runtime, request)
    }

    /// Incremental admission is only safe for execution adapters that can
    /// replay an already-streamed request after capacity preemption.
    #[cfg(test)]
    pub(crate) fn prepare_incremental(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
    ) -> Result<Option<ManagedCacheReservation>> {
        self.prepare_with_admission(
            runtime,
            txn_id,
            session,
            work,
            request,
            ManagedClaimPolicy::Incremental,
            true,
        )
    }

    fn prepare_with_admission(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
        claim: ManagedClaimPolicy,
        prefix_attach: bool,
    ) -> Result<Option<ManagedCacheReservation>> {
        let previous_claim = self
            .models
            .get(&runtime.plan.model_instance)
            .and_then(|state| state.capacity_claims.get(session))
            .cloned();
        let result = self.prepare_inner(
            runtime,
            txn_id,
            session,
            work,
            request,
            claim,
            prefix_attach,
        );
        if result.is_ok()
            && claim != ManagedClaimPolicy::Full
            && matches!(work, WorkUnit::SequenceStep { .. })
        {
            if let Some(state) = self.models.get_mut(&runtime.plan.model_instance) {
                if state.capacity_claims.contains_key(session) {
                    state.incremental_claim_sessions.insert(session.clone());
                }
            }
        }
        if result.is_err() {
            if let Some(state) = self.models.get_mut(&runtime.plan.model_instance) {
                // Every arena belongs to the same logical transaction. Also
                // catch errors from projection/prefix setup between arenas.
                for coordinator in state.coordinators.values_mut() {
                    let _ = coordinator.abort(txn_id);
                }
                match previous_claim {
                    Some(claim) => {
                        state.capacity_claims.insert(session.clone(), claim);
                    }
                    None => {
                        state.capacity_claims.remove(session);
                    }
                }
            }
        }
        result
    }

    /// DS4: drain demotion work for one model — when a device arena's
    /// pressure crosses the high watermark, move LRU committed prefix
    /// subtrees to the host pool until the low watermark, the bounded step
    /// budget, the victim supply, or the host budget runs out. Bounded and
    /// synchronous: steps run inside the manager's own critical section.
    fn run_host_offload_tick(
        policy: Option<HostOffloadPolicy>,
        state: &mut ManagedKvModelState,
        telemetry: &ManagedKvTelemetry,
    ) {
        fn pressure(coordinator: &KvCacheCoordinator) -> Option<f32> {
            let stats = coordinator.stats();
            (stats.capacity_pages > 0)
                .then(|| stats.allocated_pages as f32 / stats.capacity_pages as f32)
        }
        let Some(policy) = policy else { return };
        let Some(offload) = state.host_offload.as_mut() else {
            return;
        };
        let arena_ids: Vec<KvArenaId> = offload.arenas.keys().copied().collect();
        for arena_id in arena_ids {
            let Some(coordinator) = state.coordinators.get_mut(&arena_id) else {
                continue;
            };
            if pressure(coordinator).is_none_or(|current| current < policy.high_watermark) {
                continue;
            }
            let Some(arena) = state.runtime.arena(arena_id) else {
                continue;
            };
            let Some(index) = state.prefix_indexes.get_mut(&arena_id) else {
                continue;
            };
            let Some(arena_offload) = offload.arenas.get_mut(&arena_id) else {
                continue;
            };
            // A failure to reach the device thread demotes nothing.
            let _ = on_device_thread(arena.as_ref(), || {
                let mut steps = policy.max_in_flight_pages;
                while steps > 0 {
                    if pressure(coordinator)
                        .is_none_or(|current| current <= policy.low_watermark)
                    {
                        break;
                    }
                    match offload::demote_step(
                        coordinator,
                        index,
                        arena.as_ref(),
                        arena_offload,
                        telemetry,
                    ) {
                        Ok(DemoteOutcome::Demoted { .. }) => steps -= 1,
                        Ok(_) => break,
                        Err(_) => break,
                    }
                }
                Ok(())
            });
        }
    }

    fn prepare_inner(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        txn_id: PlanId,
        session: &SessionKey,
        work: &WorkUnit,
        request: Option<&EngineCoreRequest>,
        claim: ManagedClaimPolicy,
        prefix_attach: bool,
    ) -> Result<Option<ManagedCacheReservation>> {
        let (
            sequence_input,
            sequence_phase,
            auxiliary_state,
            realtime_cache_append,
            allow_unchanged_prefix,
            exact_target_prefix,
        ) = match work {
            WorkUnit::SequenceStep {
                input,
                auxiliary_state,
                phase,
                ..
            } => (
                Some(*input),
                Some(*phase),
                auxiliary_state.as_ref(),
                None,
                *phase == crate::engine::SequencePhase::Decode,
                false,
            ),
            WorkUnit::RealtimePush {
                max_cache_append, ..
            }
            | WorkUnit::RealtimeFinish {
                max_cache_append, ..
            } => (None, None, None, Some(*max_cache_append), true, false),
            WorkUnit::RealtimePreparation {
                auxiliary_state, ..
            } => (None, None, auxiliary_state.as_ref(), Some(0), true, false),
            WorkUnit::RealtimeCompletion { .. } => (None, None, None, Some(0), true, false),
            WorkUnit::RealtimePromptPrefill { cache_append, .. } => {
                if *cache_append == 0 {
                    return Err(Error::InvalidInput(
                        "realtime prompt prefill requires a positive exact cache append".into(),
                    ));
                }
                (None, None, None, Some(*cache_append), false, true)
            }
            WorkUnit::RealtimeDecodeContinuation {
                max_cache_append,
                auxiliary_state,
                ..
            } => {
                if *max_cache_append != 1 {
                    return Err(Error::InvalidInput(
                        "realtime decode continuation requires exactly one cache append".into(),
                    ));
                }
                (None, None, auxiliary_state.as_ref(), Some(1), false, true)
            }
            _ => return Ok(None),
        };
        if self
            .models
            .get(&runtime.plan.model_instance)
            .is_some_and(|state| state.closing)
        {
            return Err(Error::InferenceError(
                "managed KV runtime is closing".to_string(),
            ));
        }
        let namespace = managed_prefix_namespace(request, runtime, self.prefix_cache_salt)?;
        let selected_tensor_state = auxiliary_state
            .map(|spans| {
                spans
                    .iter()
                    .map(|span| {
                        let input = span.input();
                        Ok(TensorStateSelection {
                            group: span.group(),
                            clock: span.clock().clone(),
                            expected_cursor: u64::try_from(input.start).map_err(|_| {
                                Error::InvalidInput("clocked state cursor exceeds u64".to_string())
                            })?,
                            target_cursor: u64::try_from(input.end).map_err(|_| {
                                Error::InvalidInput("clocked state cursor exceeds u64".to_string())
                            })?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if realtime_cache_append.is_some()
            && runtime.tensor_state().is_some()
            && selected_tensor_state.is_none()
        {
            return Err(Error::InferenceError(
                "realtime paged reservation cannot implicitly advance tensor state".into(),
            ));
        }
        if sequence_input.is_some_and(|input| input.is_empty()) && selected_tensor_state.is_none() {
            return Err(Error::InferenceError(
                "legacy decoder-coupled tensor state cannot advance without a paged span".into(),
            ));
        }
        let needs_tensor_transaction = runtime.tensor_state().is_some()
            && selected_tensor_state
                .as_ref()
                .is_none_or(|selections| !selections.is_empty());
        if runtime.tensor_state().is_none()
            && selected_tensor_state
                .as_ref()
                .is_some_and(|selections| !selections.is_empty())
        {
            return Err(Error::InferenceError(
                "clocked state spans selected a model without a tensor-state arena".into(),
            ));
        }
        let tensor_transaction = needs_tensor_transaction
            .then(|| PhysicalStateTransactionId::new(txn_id))
            .transpose()?;
        let needs_tensor_sequence = needs_tensor_transaction
            && self
                .models
                .get(&runtime.plan.model_instance)
                .is_some_and(|state| !state.tensor_sequences.contains_key(session));
        let tensor_sequence_candidate = if needs_tensor_sequence {
            let candidate = PhysicalStateSequenceId::new(self.next_tensor_sequence)?;
            self.next_tensor_sequence = self
                .next_tensor_sequence
                .checked_add(1)
                .ok_or_else(|| Error::InferenceError("tensor-state sequence id overflow".into()))?;
            Some(candidate)
        } else {
            None
        };
        let state = self
            .models
            .get_mut(&runtime.plan.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model runtime is missing".into()))?;
        if state.runtime.plan.id != runtime.plan.id {
            return Err(Error::InferenceError(
                "request carries a stale managed KV runtime".to_string(),
            ));
        }
        // DS4: drain demotion before capacity claims so pages freed to the
        // host tier are visible to this very reservation.
        let host_offload_policy = self.host_offload_policy;
        Self::run_host_offload_tick(host_offload_policy, state, &self.telemetry);
        let host_page_ceiling = host_offload_policy
            .map(|policy| policy.max_promotion_pages)
            .unwrap_or(0);
        let installed_claim = if let Some(request) = request {
            match claim {
                ManagedClaimPolicy::Incremental if sequence_input.is_some() => {
                    ensure_incremental_capacity_claim(state, session, request, work)?
                }
                ManagedClaimPolicy::Chunked { quantum_tokens } if sequence_input.is_some() => {
                    ensure_chunked_capacity_claim(state, session, request, work, quantum_tokens)?
                }
                _ => ensure_capacity_claim(state, session, request)?,
            }
        } else {
            false
        };
        if let Err(error) = ensure_session_tables(state, session) {
            if installed_claim {
                state.capacity_claims.remove(session);
            }
            return Err(error);
        }
        let session_generation =
            state
                .session_generations
                .get(session)
                .copied()
                .ok_or_else(|| {
                    Error::InferenceError("registered managed session lost its generation".into())
                })?;

        let mut domains = Vec::with_capacity(runtime.plan.groups.len());
        let mut pending_prefixes = Vec::new();
        // DS4.3: host tails restored this transaction, committed at finalize.
        let mut pending_promotions = Vec::new();
        // DS1.2b: snapshot-sharing facts copied out of `state` so the paged
        // group loop can consult them while mutating coordinators. Clocked
        // (selected) transactions have no fork path and never share.
        let snapshot_sharing = if selected_tensor_state.is_none() {
            state.tensor_snapshots.as_ref().map(|sharing| {
                (
                    sharing.policy.clone(),
                    sharing.paged_arena,
                    sharing.interval_tokens,
                )
            })
        } else {
            None
        };
        let mut tensor_attach: Option<(u64, Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>)> =
            None;
        for group in &runtime.plan.groups {
            let domain_sequence_input = match (sequence_input, sequence_phase, request) {
                (Some(input), Some(phase), Some(request)) => {
                    Some(request.project_paged_state_input(group.domain, phase, input)?)
                }
                (input, _, _) => input,
            };
            let coordinator = state
                .coordinators
                .get_mut(&group.arena)
                .expect("resolved arena has a coordinator");
            let snapshot = coordinator
                .snapshot(session, group.domain)
                .map_err(coordinator_error)?;
            let mut target_committed_tokens = match (domain_sequence_input, realtime_cache_append) {
                (Some(input), None) => u32::try_from(input.end).map_err(|_| {
                    Error::InvalidInput("managed KV token position exceeds u32".to_string())
                })?,
                (None, Some(max_cache_append)) => snapshot
                    .committed_tokens
                    .checked_add(u32::try_from(max_cache_append).map_err(|_| {
                        Error::InvalidInput(
                            "realtime managed KV append ceiling exceeds u32".to_string(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::InvalidInput("realtime managed KV target overflow".to_string())
                    })?,
                _ => unreachable!("managed work target was authenticated above"),
            };
            if target_committed_tokens < snapshot.committed_tokens {
                abort_domains(state, txn_id, &domains);
                return Err(Error::InferenceError(
                    "scheduled KV target regressed behind the committed cache table".to_string(),
                ));
            }
            if domain_sequence_input.is_some_and(|input| input.is_empty()) {
                if target_committed_tokens != snapshot.committed_tokens {
                    abort_domains(state, txn_id, &domains);
                    return Err(Error::InferenceError(
                        "empty decoder span disagrees with the committed paged cursor".into(),
                    ));
                }
                continue;
            }
            let prefix_eligible = prefix_attach
                && snapshot.committed_tokens == 0
                && domain_sequence_input.is_some_and(|input| input.start == 0)
                // Prefix reuse is bounded by this transaction's committed
                // target below, so the first scheduler-visible chunk does not
                // need to cover the entire logical prompt.
                && request.is_some_and(|request| {
                    domain_sequence_input
                        .is_some_and(|input| input.end <= request.prompt_tokens.len())
                })
                && target_committed_tokens > 1
                && prefix_enabled_for_domain(&state.contract, group.domain);
            // A subordinate retained-session generation is a semantic restart
            // whose model handoff requires an exact context-0 physical cache.
            // Published pages remain available to unrelated fresh sessions,
            // but this restarted session must rebuild its first generation
            // span instead of attaching a prefix and beginning above zero.
            // DS1.5: an admission probe may have scheduled this span to start
            // exactly at a verified shared-prefix cursor. A fresh table whose
            // span starts there attaches directly instead of recomputing the
            // resident prefix.
            let expected_attach_cursor = request
                .and_then(|req| req.managed_prefix_cursor())
                .filter(|cursor| {
                    *cursor > 0
                        && snapshot.committed_tokens == 0
                        && domain_sequence_input
                            .is_some_and(|input| input.start == *cursor as usize)
                });
            let mut prefix_match =
                if prefix_eligible && session_generation == ManagedSessionGeneration::INITIAL {
                    if let Some(namespace) = namespace.as_ref() {
                        let reusable_tokens =
                            usize::try_from(target_committed_tokens - 1).unwrap_or(usize::MAX);
                        let host_chain = state.host_offload.as_mut().and_then(|offload| {
                            offload
                                .arenas
                                .get_mut(&group.arena)
                                .map(|arena| &mut arena.chain)
                        });
                        let prefix_index = state
                            .prefix_indexes
                            .get_mut(&group.arena)
                            .expect("resolved arena has a prefix index");
                        managed_prefix_lookup(
                            prefix_index,
                            host_chain,
                            group.page_tokens,
                            namespace,
                            &request
                                .expect("prefix namespace requires a request")
                                .prompt_tokens[..reusable_tokens],
                            host_page_ceiling,
                        )?
                    } else {
                        self.telemetry.record_prefix_rejection();
                        Default::default()
                    }
                } else if let Some(cursor) = expected_attach_cursor {
                    if let Some(namespace) = namespace.as_ref() {
                        let reusable_tokens = cursor as usize;
                        let host_chain = state.host_offload.as_mut().and_then(|offload| {
                            offload
                                .arenas
                                .get_mut(&group.arena)
                                .map(|arena| &mut arena.chain)
                        });
                        let prefix_index = state
                            .prefix_indexes
                            .get_mut(&group.arena)
                            .expect("resolved arena has a prefix index");
                        managed_prefix_lookup(
                            prefix_index,
                            host_chain,
                            group.page_tokens,
                            namespace,
                            &request
                                .expect("prefix namespace requires a request")
                                .prompt_tokens[..reusable_tokens],
                            host_page_ceiling,
                        )?
                    } else {
                        Default::default()
                    }
                } else {
                    Default::default()
                };
            if let Some(cursor) = expected_attach_cursor {
                // The attach must reconcile the same snapshot boundary the
                // probe verified; a shorter realized cursor means the shared
                // state was evicted under us and the scheduled span is stale.
                if let Some((policy, paged_arena, _interval)) = &snapshot_sharing {
                    if *paged_arena == group.arena && policy.paged_domain == group.domain {
                        if let Some(sharing) = state.tensor_snapshots.as_mut() {
                            match reconcile_tensor_snapshot_attach(
                                sharing,
                                group.page_tokens,
                                &mut prefix_match,
                                needs_tensor_sequence,
                                &self.telemetry,
                            ) {
                                Ok(Some(attach)) => tensor_attach = Some(attach),
                                Ok(None) => {}
                                Err(error) => {
                                    abort_domains(state, txn_id, &domains);
                                    return Err(error);
                                }
                            }
                        }
                    }
                }
                if prefix_match.reused_tokens != cursor {
                    abort_domains(state, txn_id, &domains);
                    return Err(Error::Backpressure(format!(
                        "{MANAGED_PREFIX_CURSOR_LOST}: admission probed cursor {cursor} but the index now offers {}",
                        prefix_match.reused_tokens
                    )));
                }
            }
            // DS1.2b: tensor domains may only follow a paged attach when a
            // committed snapshot exists at the attach cursor. Walk the matched
            // chain back to the nearest snapshot-backed boundary, falling back
            // to zero reuse when none exists or the session cannot fork.
            let mut publication_start_tokens;
            if prefix_eligible {
                if let Some((policy, paged_arena, interval_tokens)) = &snapshot_sharing {
                    if *paged_arena == group.arena && policy.paged_domain == group.domain {
                        let pre_tensor_reused = prefix_match.reused_tokens;
                        if let Some(sharing) = state.tensor_snapshots.as_mut() {
                            match reconcile_tensor_snapshot_attach(
                                sharing,
                                group.page_tokens,
                                &mut prefix_match,
                                needs_tensor_sequence,
                                &self.telemetry,
                            ) {
                                Ok(Some(attach)) => tensor_attach = Some(attach),
                                Ok(None) => {}
                                Err(error) => {
                                    abort_domains(state, txn_id, &domains);
                                    return Err(error);
                                }
                            }
                        }
                        // Nothing attached: align the fresh commit cursor to the
                        // declared snapshot interval so the committed tensor
                        // state can be published at the same boundary the paged
                        // pages end on. The residual tokens are served by the
                        // next prefill chunk. An attached session keeps its
                        // exact target and simply commits the residual span.
                        //
                        // The reduction is only sound when the scheduled input
                        // span already ends at (or below) the boundary — the
                        // scheduler's first-chunk alignment guarantees that.
                        // Truncating a commit target below the span the
                        // executor is about to append would leave the model
                        // writing pages past its block table.
                        if tensor_attach.is_none() && prefix_match.reused_tokens == 0 {
                            let interval = u64::from(*interval_tokens);
                            let target = u64::from(target_committed_tokens);
                            let boundary = target - target % interval;
                            let span_fits_boundary = domain_sequence_input
                                .map(|input| {
                                    u64::try_from(input.end).is_ok_and(|end| end <= boundary)
                                })
                                .unwrap_or(false);
                            if span_fits_boundary && boundary > 0 && boundary < target {
                                target_committed_tokens =
                                    u32::try_from(boundary).map_err(|_| {
                                        Error::InvalidInput(
                                            "managed KV snapshot boundary exceeds u32".into(),
                                        )
                                    })?;
                            }
                        }
                        // Pages the lookup matched but the reconcile dropped are
                        // already published by another session under the same
                        // digests; republishing them from fresh blocks would
                        // bind one digest to two blocks. Publish only pages the
                        // lookup did not already see.
                        publication_start_tokens =
                            pre_tensor_reused.max(prefix_match.reused_tokens);
                    } else {
                        publication_start_tokens = prefix_match.reused_tokens;
                    }
                } else {
                    publication_start_tokens = prefix_match.reused_tokens;
                }
            } else {
                publication_start_tokens = prefix_match.reused_tokens;
            }
            // DS4: a host tail means pages from the device boundary onward
            // bind into the device index at this transaction's commit — the
            // restored ones and any recomputed ones alike — so publication
            // starts at the device-resident boundary instead of the execution
            // cursor (finalized after the restore below).
            let host_tail_boundary = prefix_match
                .host_tail
                .as_ref()
                .map(|tail| tail.device_end_tokens);
            let sliding_window = sliding_window_for_domain(&state.contract, group.domain)?;
            // The earliest query of this span sits at `start` and still attends
            // the `window` keys ending at itself, so the table must keep keys
            // from `start + 1 - window`. Keeping only `target - window` drops
            // the lookback of every query but the last in a multi-token span
            // (chunked prefill, speculative verification). A single-token step
            // is unchanged, and the next step advances the window again.
            let target_window_start = sliding_window
                .map(|window| {
                    let start = domain_sequence_input
                        .map(|input| u32::try_from(input.start).unwrap_or(u32::MAX))
                        .unwrap_or(snapshot.committed_tokens);
                    target_committed_tokens
                        .saturating_sub(window)
                        .min(start.saturating_add(1).saturating_sub(window))
                })
                .unwrap_or(0);
            let established_window_table = sliding_window.is_some()
                && snapshot.groups.iter().any(|table| table.group == group.id);
            let reserve_request = if established_window_table {
                None
            } else {
                let reservation = reservation_for_group(
                    group.id,
                    group.page_tokens,
                    &snapshot,
                    target_committed_tokens,
                    &prefix_match.blocks,
                )?;
                Some(KvReserveRequest {
                    txn_id,
                    expected: snapshot.clone(),
                    target_committed_tokens,
                    target_window_start: 0,
                    groups: vec![reservation],
                })
            };
            let reserve_once = |coordinator: &mut KvCacheCoordinator| {
                if established_window_table {
                    coordinator.reserve_window(KvWindowReserveRequest {
                        txn_id,
                        expected: snapshot.clone(),
                        target_committed_tokens,
                        target_window_start,
                        page_tokens: group.page_tokens,
                    })
                } else {
                    let request = reserve_request
                        .as_ref()
                        .expect("non-window reservation exists");
                    match coordinator.reserve(request.clone()) {
                        Err(KvCoordinatorError::WriteConflict) => {
                            let mut copy_on_write = request.clone();
                            for intent in &mut copy_on_write.groups[0].blocks {
                                if let KvBlockIntent::Writable(source) = *intent {
                                    *intent = KvBlockIntent::CopyOnWrite(source);
                                }
                            }
                            coordinator.reserve(copy_on_write)
                        }
                        result => result,
                    }
                }
            };
            let mut reserved = reserve_once(coordinator);
            while matches!(reserved, Err(KvCoordinatorError::Capacity)) {
                let protected = prefix_match.blocks.iter().copied().collect::<HashSet<_>>();
                let evicted = state
                    .prefix_indexes
                    .get_mut(&group.arena)
                    .expect("resolved arena has a prefix index")
                    .evict_lru_excluding(coordinator, &protected)
                    .map_err(prefix_error)?;
                if evicted.is_empty() {
                    break;
                }
                reserved = reserve_once(coordinator);
            }
            if let Err(error) = reserved {
                abort_domains(state, txn_id, &domains);
                if matches!(error, KvCoordinatorError::Capacity) {
                    return Err(Error::Backpressure(
                        "managed KV arena has no reservable pages".to_string(),
                    ));
                }
                return Err(coordinator_error(error));
            }
            let prepared = match coordinator.prepare(txn_id) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(coordinator_error(error));
                }
            };
            let required_resident_pages = prepared
                .provisional_groups
                .iter()
                .flat_map(|group| group.blocks.iter())
                .map(|block| block.index.saturating_add(1))
                .max()
                .unwrap_or(0);
            let arena = runtime
                .arena(group.arena)
                .expect("resolved arena allocated");
            let growth_result = (|| -> Result<()> {
                if let Some(growth) = arena.plan_resident_growth(required_resident_pages)? {
                    let added_bytes = group
                        .bytes_per_page
                        .checked_mul(u64::from(growth.added_pages()))
                        .ok_or_else(|| {
                            Error::Overloaded("managed KV growth byte total overflow".into())
                        })?;
                    let final_materialized =
                        state.materialized_resources.checked_add(ResourceVector {
                            device_bytes: ResourceAmount::Known(added_bytes),
                            ..ResourceVector::zero()
                        })?;
                    // Replacing a contiguous Candle tensor keeps the old arena
                    // alive while allocating the complete target arena. Reserve
                    // that admission-only peak before touching device memory,
                    // then shrink the lease to the final resident footprint.
                    let target_bytes = group
                        .bytes_per_page
                        .checked_mul(u64::from(growth.target_pages))
                        .ok_or_else(|| {
                            Error::Overloaded("managed KV growth peak byte total overflow".into())
                        })?;
                    let allocation_peak =
                        state.materialized_resources.checked_add(ResourceVector {
                            device_bytes: ResourceAmount::Known(target_bytes),
                            ..ResourceVector::zero()
                        })?;
                    if let Some(lease) = state.resource_lease.as_ref() {
                        if !allocation_peak.fits_within(lease.resources()) {
                            return Err(Error::InferenceError(
                                "managed CUDA growth exceeds its load-time replacement reservation"
                                    .into(),
                            ));
                        }
                    }
                    on_device_thread(arena.as_ref(), || arena.grow_resident_pages(growth))?;
                    state.allocation_ledger.reconcile_group_receipt(
                        runtime.allocation_plan(),
                        crate::kv::v2::StateGroupId::new(group.id.get()),
                        group.domain,
                        AllocationReceipt {
                            requested_owned_bytes: added_bytes,
                            committed_owned_bytes: added_bytes,
                            allocator_overhead_bytes: 0,
                            residency: ResidencyMeasurement::Unknown,
                        },
                    )?;
                    if let Some(lease) = state.resource_lease.as_ref() {
                        lease.record_materialized_usage(final_materialized.checked_add(
                            tensor_state_resources(
                                state.runtime.plan.backend,
                                state.tensor_materialized_bytes,
                            )?,
                        )?)?;
                    }
                    state.materialized_resources = final_materialized;
                    self.telemetry.record_backing_allocation();
                }
                Ok(())
            })();
            if let Err(error) = growth_result {
                let _ = coordinator.abort(txn_id);
                abort_domains(state, txn_id, &domains);
                return Err(error);
            }
            let fresh = &prepared.fresh_blocks;
            if !fresh.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = on_device_thread(arena.as_ref(), || {
                    arena.zero_pages(fresh).and_then(|fence| fence.wait())
                }) {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
                self.telemetry.record_zero(fresh.len());
            }
            if !prepared.page_copies.is_empty() {
                let arena = runtime
                    .arena(group.arena)
                    .expect("resolved arena allocated");
                if let Err(error) = on_device_thread(arena.as_ref(), || {
                    arena
                        .copy_pages(&prepared.page_copies)
                        .and_then(|fence| fence.wait())
                }) {
                    let _ = coordinator.abort(txn_id);
                    abort_domains(state, txn_id, &domains);
                    return Err(error);
                }
                self.telemetry.record_copy(prepared.page_copies.len());
                if !prefix_match.blocks.is_empty() {
                    self.telemetry
                        .record_prefix_copy_on_write(prepared.page_copies.len());
                }
            }
            // DS4.3: restore the host-resident tail of the matched prefix into
            // the fresh pages this transaction reserved for their span. A page
            // that fails to restore truncates the promotion (the failing entry
            // and its host descendants are purged); with an admission cursor
            // the shortened match degrades to the cursor-lost re-plan, without
            // one the executor simply recomputes the truncated span.
            if let Some(tail) = prefix_match.host_tail.take() {
                let promoted = {
                    let arena = runtime
                        .arena(group.arena)
                        .expect("resolved arena allocated");
                    match state
                        .host_offload
                        .as_mut()
                        .and_then(|offload| offload.arenas.get_mut(&group.arena))
                    {
                        Some(arena_offload) => {
                            let table = prepared
                                .provisional_groups
                                .iter()
                                .find(|table| table.group == group.id);
                            promote_host_tail(
                                arena.as_ref(),
                                arena_offload,
                                group.arena,
                                group.page_tokens,
                                table,
                                tail,
                            )
                        }
                        // The host tier vanished mid-prepare: nothing was
                        // restored, so the match falls back to its device head.
                        None => PromotedTail {
                            restored_tokens: tail.device_end_tokens,
                            record: None,
                        },
                    }
                };
                if promoted.restored_tokens < prefix_match.reused_tokens {
                    if expected_attach_cursor
                        .is_some_and(|cursor| promoted.restored_tokens != cursor)
                    {
                        let _ = coordinator.abort(txn_id);
                        abort_domains(state, txn_id, &domains);
                        return Err(Error::Backpressure(format!(
                            "{MANAGED_PREFIX_CURSOR_LOST}: promotion restored cursor {} but admission probed {}",
                            promoted.restored_tokens, prefix_match.reused_tokens
                        )));
                    }
                    prefix_match.reused_tokens = promoted.restored_tokens;
                }
                if let Some(record) = promoted.record {
                    pending_promotions.push(record);
                }
            }
            let execution_start_tokens = snapshot.committed_tokens.max(prefix_match.reused_tokens);
            publication_start_tokens = match host_tail_boundary {
                Some(device_end_tokens) => publication_start_tokens.min(device_end_tokens),
                None => publication_start_tokens.max(execution_start_tokens),
            };
            domains.push(ManagedCacheDomainReservation {
                arena: group.arena,
                domain: group.domain,
                expected_version: prepared.expected.version,
                expected_committed_tokens: prepared.expected.committed_tokens,
                execution_start_tokens,
                target_committed_tokens: prepared.target_committed_tokens,
                target_window_start: prepared.target_window_start,
                first_page_offset: prepared.target_window_start % group.page_tokens,
                provisional_groups: prepared.provisional_groups,
                writable_blocks: prepared.writable_blocks,
            });
            // Phase 5a: a request declaring its stable-history boundary also
            // publishes the pages of every later prefill span inside its
            // prompt, so a conversation's chain keeps growing turn by turn
            // instead of stopping at the first chunk.
            let declared_boundaries = request
                .map(|request| {
                    [
                        request.managed_preamble_boundary(),
                        request.managed_snapshot_boundary(),
                    ]
                })
                .unwrap_or_default();
            let declared_boundary = declared_boundaries.iter().any(Option::is_some);
            let boundary_publication = declared_boundary
                && session_generation == ManagedSessionGeneration::INITIAL
                && target_committed_tokens > 1
                && request.is_some_and(|request| {
                    domain_sequence_input
                        .is_some_and(|input| input.end <= request.prompt_tokens.len())
                })
                && prefix_enabled_for_domain(&state.contract, group.domain);
            if prefix_eligible || expected_attach_cursor.is_some() || boundary_publication {
                let Some(namespace) = namespace.as_ref() else {
                    continue;
                };
                let publications = prefix_publications(
                    namespace,
                    &request
                        .expect("prefix namespace requires a request")
                        .prompt_tokens,
                    group.page_tokens,
                    publication_start_tokens,
                    target_committed_tokens,
                    domains
                        .last()
                        .expect("domain reservation was just appended"),
                    group.id,
                )?;
                if !publications.is_empty() {
                    // Without a declared boundary keep the DS1.2b rule (any
                    // aligned commit of an eligible span). With them, only a
                    // span ending on a boundary's interval floor (first turn
                    // or history) snapshots: each snapshot is the full
                    // hybrid state.
                    let snapshot = match (declared_boundary, &snapshot_sharing) {
                        (false, _) => true,
                        (true, Some((_, _, interval))) if *interval > 0 => {
                            declared_boundaries.iter().flatten().any(|boundary| {
                                target_committed_tokens == boundary - boundary % *interval
                            })
                        }
                        (true, _) => false,
                    };
                    pending_prefixes.push(PendingPrefixCommit {
                        arena: group.arena,
                        page_tokens: group.page_tokens,
                        publications,
                        snapshot,
                    });
                }
            }
        }
        if !pending_prefixes.is_empty()
            && state
                .pending_prefixes
                .insert(txn_id, pending_prefixes)
                .is_some()
        {
            abort_domains(state, txn_id, &domains);
            return Err(Error::InferenceError(
                "managed KV transaction duplicated pending prefix publication".into(),
            ));
        }
        if !pending_promotions.is_empty() {
            state.pending_promotions.insert(txn_id, pending_promotions);
        }
        let clocked_state = if needs_tensor_transaction {
            let arena = runtime
                .tensor_state()
                .expect("tensor transaction requires an arena");
            let transaction = tensor_transaction.expect("tensor arena has a transaction");
            let (sequence, newly_registered) =
                if let Some(sequence) = state.tensor_sequences.get(session).copied() {
                    (sequence, false)
                } else {
                    let sequence = tensor_sequence_candidate.expect("tensor arena has a candidate");
                    if let Err(error) = arena.register(sequence) {
                        abort_domains(state, txn_id, &domains);
                        state.pending_prefixes.remove(&txn_id);
                        state.pending_promotions.remove(&txn_id);
                        return Err(error);
                    }
                    state.tensor_sequences.insert(session.clone(), sequence);
                    (sequence, true)
                };
            // DS1.2b: attach-by-fork. Seed the freshly registered sequence with
            // the committed snapshot (read -> begin -> stage -> commit at the
            // fork cursor, the DS1.1 prototype recipe) so the executor's
            // tensor updates stage on top of the attach cursor exactly as the
            // paged window does.
            if let Some((fork_cursor, snapshot_domains)) = tensor_attach.take() {
                if !newly_registered {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    state.pending_promotions.remove(&txn_id);
                    return Err(Error::InferenceError(
                        "managed tensor snapshot attach requires a fresh sequence".into(),
                    ));
                }
                let fork_result = (|| -> Result<()> {
                    let fork_transaction = self.next_tensor_fork_transaction;
                    self.next_tensor_fork_transaction =
                        fork_transaction.checked_add(1).ok_or_else(|| {
                            Error::InferenceError(
                                "tensor fork transaction space is exhausted".into(),
                            )
                        })?;
                    let seed = PhysicalStateTransactionId::new(u64::MAX - fork_transaction)?;
                    let staged = (|| -> Result<()> {
                        arena.begin(seed, sequence)?;
                        for (domain, components) in snapshot_domains.iter() {
                            arena.stage_replace(
                                seed,
                                *domain,
                                0,
                                fork_cursor,
                                components.iter().cloned().collect(),
                            )?;
                        }
                        // A fresh sequence has no committed state to retire.
                        arena.commit(seed, fork_cursor).map(drop)
                    })();
                    if staged.is_err() {
                        let _ = arena.abort(seed);
                    }
                    staged
                })();
                if let Err(error) = fork_result {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    state.pending_promotions.remove(&txn_id);
                    state.tensor_sequences.remove(session);
                    release_tensor_sequence(state, arena, sequence).map_err(|release_error| {
                        Error::InferenceError(format!(
                            "tensor snapshot fork failed ({error}); newly registered sequence rollback also failed: {release_error}"
                        ))
                    })?;
                    return Err(error);
                }
                // The fork commits snapshot tensors that already exist.
                reconcile_tensor_materialization(state);
                self.telemetry.record_tensor_snapshot_attach();
            }
            let managed_reservation = if let Some(selections) = selected_tensor_state.as_ref() {
                ManagedClockedStateReservation::selected(
                    runtime.plan().model_instance,
                    sequence.get(),
                    selections.clone().into(),
                )
            } else {
                ManagedClockedStateReservation::legacy(
                    runtime.plan().model_instance,
                    sequence.get(),
                )
            };
            let managed_reservation = match managed_reservation {
                Ok(reservation) => reservation,
                Err(error) => {
                    abort_domains(state, txn_id, &domains);
                    state.pending_prefixes.remove(&txn_id);
                    state.pending_promotions.remove(&txn_id);
                    if newly_registered {
                        state.tensor_sequences.remove(session);
                        release_tensor_sequence(state, arena, sequence)?;
                    }
                    return Err(error);
                }
            };
            let begin = if let Some(selections) = selected_tensor_state.as_ref() {
                arena.begin_selected(transaction, sequence, selections)
            } else {
                arena.begin(transaction, sequence)
            };
            if let Err(error) = begin {
                abort_domains(state, txn_id, &domains);
                state.pending_prefixes.remove(&txn_id);
                state.pending_promotions.remove(&txn_id);
                if newly_registered {
                    state.tensor_sequences.remove(session);
                    release_tensor_sequence(state, arena, sequence).map_err(|release_error| {
                        Error::InferenceError(format!(
                            "tensor transaction admission failed ({error}); newly registered sequence rollback also failed: {release_error}"
                        ))
                    })?;
                }
                return Err(error);
            }
            Some(managed_reservation)
        } else {
            None
        };
        if domains.is_empty() && clocked_state.is_none() {
            return Ok(None);
        }
        if exact_target_prefix && !state.exact_target_transactions.insert(txn_id) {
            abort_reservation(
                state,
                &ManagedCacheReservation {
                    txn_id,
                    session: session.clone(),
                    session_generation,
                    domains: domains.clone(),
                    clocked_state: clocked_state.clone(),
                    allow_unchanged_prefix,
                },
            );
            return Err(Error::InferenceError(
                "managed exact-target transaction identity was reused".into(),
            ));
        }
        Ok(Some(ManagedCacheReservation {
            txn_id,
            session: session.clone(),
            session_generation,
            domains,
            clocked_state,
            allow_unchanged_prefix,
        }))
    }

    pub(crate) fn finalize(
        &mut self,
        reservation: &ManagedCacheReservation,
        receipt: Option<&ManagedCacheReceipt>,
        commit: bool,
    ) -> Result<()> {
        let model_instance = if let Some(domain) = reservation.domains.first() {
            domain.arena.model_instance
        } else if let Some(clocked) = reservation.clocked_state.as_ref() {
            clocked.model_instance()
        } else {
            return Err(Error::InferenceError(
                "managed reservation contains no paged or clocked state".into(),
            ));
        };
        let state = self
            .models
            .get_mut(&model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model state is missing".into()))?;
        if !commit {
            abort_reservation(state, reservation);
            self.telemetry.record_abort();
            return Ok(());
        }
        if let Err(error) = validate_reservation_session_generation(state, reservation) {
            abort_reservation(state, reservation);
            return Err(error);
        }
        if let Some(clocked) = reservation.clocked_state.as_ref() {
            let sequence = PhysicalStateSequenceId::new(clocked.sequence())?;
            if clocked.model_instance() != model_instance
                || state.tensor_sequences.get(&reservation.session) != Some(&sequence)
            {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "clocked-state reservation crossed its model/session sequence fence".into(),
                ));
            }
        }
        let receipt = match receipt {
            Some(receipt) => receipt,
            None => {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "committing managed KV row omitted its write receipt".into(),
                ));
            }
        };
        if &receipt.reservation != reservation {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV receipt crossed a row transaction fence".to_string(),
            ));
        }
        if let Err(error) = receipt.validate() {
            abort_reservation(state, reservation);
            return Err(error);
        }

        let accepted_prefix = receipt.accepted_prefix();
        if state
            .exact_target_transactions
            .contains(&reservation.txn_id)
            && reservation.domains.iter().any(|domain| {
                accepted_prefix.unwrap_or(domain.target_committed_tokens)
                    != domain.target_committed_tokens
            })
        {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV exact-target reservation rejected a partial prefix".into(),
            ));
        }
        // DS4.3: host tails restored at prepare, reclaimed at this commit.
        let promotions = state
            .pending_promotions
            .get(&reservation.txn_id)
            .cloned()
            .unwrap_or_default();
        let mut resolved_domains = Vec::with_capacity(reservation.domains.len());
        for domain in &reservation.domains {
            let Some(written) = receipt
                .domains
                .iter()
                .find(|receipt| receipt.arena == domain.arena && receipt.domain == domain.domain)
            else {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "managed KV receipt omitted a cache domain".into(),
                ));
            };
            let group = state
                .runtime
                .plan
                .groups
                .iter()
                .find(|group| group.arena == domain.arena && group.domain == domain.domain)
                .ok_or_else(|| {
                    Error::InferenceError(
                        "managed KV reservation lost its authoritative page geometry".into(),
                    )
                });
            let group = match group {
                Ok(group) => group,
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(error);
                }
            };
            let committed_tokens = accepted_prefix.unwrap_or(domain.target_committed_tokens);
            if committed_tokens < domain.execution_start_tokens
                || committed_tokens > domain.target_committed_tokens
            {
                abort_reservation(state, reservation);
                return Err(Error::InferenceError(
                    "managed KV accepted prefix is outside one domain reservation".into(),
                ));
            }
            let target_window_start = if accepted_prefix.is_some() {
                match sliding_window_for_domain(&state.contract, domain.domain) {
                    Ok(Some(window)) => committed_tokens
                        .saturating_sub(window)
                        .min(domain.expected_committed_tokens),
                    Ok(None) => 0,
                    Err(error) => {
                        abort_reservation(state, reservation);
                        return Err(error);
                    }
                }
            } else {
                domain.target_window_start
            };
            // DS4.3: pages restored from the host tier were filled by the
            // promotion copy rather than the executor's backend writes; the
            // committed write receipt covers both. Restored pages the accepted
            // cursor truncates away are left out — their host entries stay
            // authoritative and only the surviving span is reclaimed.
            let mut written_blocks = written.written_blocks.clone();
            if let Some(promotion) = promotions
                .iter()
                .find(|promotion| promotion.arena == domain.arena)
            {
                for (index, block) in promotion.blocks.iter().enumerate() {
                    let page_end = u64::from(promotion.device_end_tokens)
                        + u64::from(promotion.page_tokens).saturating_mul(index as u64 + 1);
                    if page_end <= u64::from(committed_tokens) && !written_blocks.contains(block) {
                        written_blocks.push(*block);
                    }
                }
            }
            resolved_domains.push((
                domain,
                written_blocks,
                committed_tokens,
                target_window_start,
                group.page_tokens,
            ));
        }

        // Mark every live transaction written. This changes no table/index
        // ownership and is rolled back by abort if any later validation fails.
        for (domain, written_blocks, committed_tokens, target_window_start, page_tokens) in
            &resolved_domains
        {
            let receipt = KvWriteReceipt {
                txn_id: reservation.txn_id,
                committed_tokens: *committed_tokens,
                written_blocks: written_blocks.clone(),
            };
            let coordinator = state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator");
            let completed = if accepted_prefix.is_some() {
                coordinator.complete_write_prefix(receipt, *target_window_start, *page_tokens)
            } else {
                coordinator.complete_write(receipt)
            };
            if let Err(error) = completed {
                abort_reservation(state, reservation);
                return Err(coordinator_error(error));
            }
        }
        let mut pending = state
            .pending_prefixes
            .get(&reservation.txn_id)
            .cloned()
            .unwrap_or_default();
        let mut staged = Vec::<(
            KvArenaId,
            KvCoordinatorCommitPlan,
            Option<StagedPrefixCommit>,
        )>::with_capacity(reservation.domains.len());
        // DS1.2b: snapshot boundary (chain digest, cursor end) captured from
        // the accepted paged publications of the sharing arena.
        let mut snapshot_boundary: Option<([u8; 32], u64)> = None;
        // DS1.2b: committed cursor of a legacy tensor transaction, set after
        // `arena.commit` succeeds.
        let mut tensor_commit_cursor: Option<u64> = None;
        for domain in &reservation.domains {
            let prefix = if let Some(index) = pending
                .iter()
                .position(|publication| publication.arena == domain.arena)
            {
                let mut publication = pending.swap_remove(index);
                let committed_tokens = accepted_prefix.unwrap_or(domain.target_committed_tokens);
                publication.publications.retain(|candidate| {
                    candidate
                        .key
                        .start_position
                        .checked_add(candidate.key.tokens.len() as u64)
                        .is_some_and(|end| end <= u64::from(committed_tokens))
                });
                if publication.publications.is_empty() {
                    None
                } else {
                    // DS1.2b: the snapshot boundary is the accepted page whose
                    // end equals the committed cursor; capture it for the
                    // tensor publication after the state commits.
                    if let Some(sharing) = state.tensor_snapshots.as_ref() {
                        if sharing.paged_arena == domain.arena && publication.snapshot {
                            let last = publication.publications.last().expect("non-empty");
                            if let Some(end) = last
                                .key
                                .start_position
                                .checked_add(last.key.tokens.len() as u64)
                            {
                                snapshot_boundary = Some((last.key.digest(), end));
                            }
                        }
                    }
                    let staged_prefix = state
                        .prefix_indexes
                        .get(&domain.arena)
                        .expect("reservation arena has a prefix index")
                        .stage_transaction(publication.page_tokens, &publication.publications);
                    Some(match staged_prefix {
                        Ok(staged) => staged,
                        Err(error) => {
                            abort_reservation(state, reservation);
                            return Err(prefix_error(error));
                        }
                    })
                }
            } else {
                None
            };
            let coordinator = state
                .coordinators
                .get_mut(&domain.arena)
                .expect("reservation arena has a coordinator");
            let commit = coordinator.stage_commit_with_prefix_updates(
                reservation.txn_id,
                prefix
                    .as_ref()
                    .map(StagedPrefixCommit::retained)
                    .unwrap_or(&[]),
                prefix
                    .as_ref()
                    .map(StagedPrefixCommit::released)
                    .unwrap_or(&[]),
            );
            match commit {
                Ok(commit) => staged.push((domain.arena, commit, prefix)),
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(coordinator_error(error));
                }
            }
        }
        if !pending.is_empty() {
            abort_reservation(state, reservation);
            return Err(Error::InferenceError(
                "managed KV transaction contains a prefix publication for an unknown domain".into(),
            ));
        }
        if let Some(clocked_state) = reservation.clocked_state.as_ref() {
            // Every failure below must abort the reservation; returning early
            // with `?` would leave the coordinator transaction `Written` with
            // its pins and reservations held, which later blocks unload.
            let committed = (|| {
                let arena = state.runtime.tensor_state().ok_or_else(|| {
                    Error::InferenceError("tensor-state reservation lost its physical arena".into())
                })?;
                let transaction = PhysicalStateTransactionId::new(reservation.txn_id)?;
                if clocked_state.selections().is_some() {
                    let completion = receipt
                        .clocked_state()
                        .expect("validated selected receipt has a completion")
                        .completion();
                    return arena.commit_selected(transaction, completion);
                }
                let target_cursor = reservation
                    .domains
                    .first()
                    .map(|domain| {
                        u64::from(accepted_prefix.unwrap_or(domain.target_committed_tokens))
                    })
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "legacy tensor transaction requires a paged cursor".into(),
                        )
                    })?;
                if reservation.domains.iter().any(|domain| {
                    u64::from(accepted_prefix.unwrap_or(domain.target_committed_tokens))
                        != target_cursor
                }) {
                    return Err(Error::InferenceError(
                        "one managed state transaction resolved divergent domain cursors".into(),
                    ));
                }
                arena.commit(transaction, target_cursor)
            })();
            let retired = match committed {
                Ok(retired) => retired,
                Err(error) => {
                    abort_reservation(state, reservation);
                    return Err(error);
                }
            };
            reconcile_tensor_materialization(state);
            drop(retired);
            if clocked_state.selections().is_none() {
                // DS1.2b: a legacy transaction committed every shared tensor
                // domain at one cursor; record it for the snapshot publication
                // below. Clocked completions publish only selected groups and
                // never share.
                tensor_commit_cursor = reservation.domains.first().map(|domain| {
                    u64::from(accepted_prefix.unwrap_or(domain.target_committed_tokens))
                });
            }
        }
        // Every fallible operation has succeeded. Applying these plans cannot
        // fail, and the engine state lock prevents an interleaving mutation.
        for (arena, commit, prefix) in staged {
            state
                .coordinators
                .get_mut(&arena)
                .expect("staged arena has a coordinator")
                .apply_staged_commit(commit);
            if let Some(prefix) = prefix {
                state
                    .prefix_indexes
                    .get_mut(&arena)
                    .expect("staged arena has a prefix index")
                    .apply_staged(prefix);
            }
        }
        // DS4.3: the transaction committed, so its restored pages are
        // device-resident and published into the prefix index; reclaim the
        // host entries they replace. An entry already evicted from the host
        // tier is simply gone.
        for promotion in state
            .pending_promotions
            .remove(&reservation.txn_id)
            .unwrap_or_default()
        {
            if let Some(arena_offload) = state
                .host_offload
                .as_mut()
                .and_then(|offload| offload.arenas.get_mut(&promotion.arena))
            {
                for digest in &promotion.digests {
                    if let Some(slot) = arena_offload.chain.remove(digest) {
                        let _ = arena_offload.pool.release_slot(slot);
                    }
                }
            }
            self.telemetry
                .record_promotion(promotion.digests.len(), promotion.latency_ns);
        }
        state.pending_prefixes.remove(&reservation.txn_id);
        state.exact_target_transactions.remove(&reservation.txn_id);
        // DS1.2b: publish the committed tensor snapshot when the commit cursor
        // is the aligned boundary of the accepted page chain. This runs after
        // every fallible operation, so a skipped publication only loses reuse
        // and never leaves committed state without its paged chain.
        if let Some((digest, boundary)) = snapshot_boundary {
            if tensor_commit_cursor.is_some_and(|cursor| cursor == boundary) {
                let published = publish_tensor_snapshot(state, reservation, digest, boundary);
                match published {
                    Ok(true) => self.telemetry.record_tensor_snapshot_publish(),
                    Ok(false) => {}
                    Err(error) => tracing::warn!(
                        error = %error,
                        "committed tensor snapshot publication skipped"
                    ),
                }
            }
        }
        self.telemetry.record_commit();
        Ok(())
    }

    pub(crate) fn release_session(&mut self, session: &SessionKey) -> Result<()> {
        for state in self.models.values_mut() {
            if !state.registered_sessions.contains(session) {
                continue;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .validate_table_release(session, group.domain)
                    .map_err(coordinator_error)?;
            }
            if let Some(sequence) = state.tensor_sequences.get(session).copied() {
                state
                    .runtime
                    .tensor_state()
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "registered tensor-state sequence lost its arena".into(),
                        )
                    })?
                    .validate_release(sequence)?;
            }
            for group in &state.runtime.plan.groups {
                state
                    .coordinators
                    .get_mut(&group.arena)
                    .expect("resolved arena has a coordinator")
                    .release_table(session, group.domain)
                    .map_err(coordinator_error)?;
            }
            if let Some(sequence) = state.tensor_sequences.remove(session) {
                let arena = state.runtime.tensor_state().cloned().ok_or_else(|| {
                    Error::InferenceError("registered tensor-state sequence lost its arena".into())
                })?;
                release_tensor_sequence(state, &arena, sequence)?;
            }
            state.registered_sessions.remove(session);
            state.session_generations.remove(session);
            state.capacity_claims.remove(session);
            state.incremental_claim_sessions.remove(session);
        }
        Ok(())
    }

    /// Replace every paged table for one exact retained session by an empty
    /// table in a new subordinate generation. The request/session identity and
    /// its capacity claim remain owned throughout the reset, so another row
    /// cannot consume the admitted logical capacity between attempts.
    pub(crate) fn reset_session_generation(
        &mut self,
        runtime: &ManagedKvModelRuntime,
        session: &SessionKey,
        expected: ManagedSessionGeneration,
    ) -> Result<ManagedSessionGeneration> {
        let state = self
            .models
            .get_mut(&runtime.plan.model_instance)
            .ok_or_else(|| Error::InferenceError("managed KV model state is missing".into()))?;
        if state.closing || state.runtime.plan.id != runtime.plan.id {
            return Err(Error::InferenceError(
                "managed KV reset carries a closing or stale model runtime".into(),
            ));
        }
        if runtime.tensor_state().is_some() {
            return Err(Error::InferenceError(
                "managed KV session reset requires a paged-only runtime".into(),
            ));
        }
        if !state.registered_sessions.contains(session) {
            return Err(Error::InferenceError(
                "managed KV reset requires a registered session".into(),
            ));
        }
        let current = state
            .session_generations
            .get(session)
            .copied()
            .ok_or_else(|| {
                Error::InferenceError("registered managed session lost its generation".into())
            })?;
        if current != expected {
            return Err(Error::InferenceError(format!(
                "managed KV reset expected session generation {}, found {}",
                expected.get(),
                current.get()
            )));
        }
        let next = current.next()?;
        let mut staged = Vec::<(KvArenaId, KvCoordinatorTableResetPlan)>::with_capacity(
            state.runtime.plan.groups.len(),
        );
        for group in &state.runtime.plan.groups {
            let coordinator = state
                .coordinators
                .get(&group.arena)
                .expect("resolved arena has a coordinator");
            let snapshot = coordinator
                .snapshot(session, group.domain)
                .map_err(coordinator_error)?;
            let next_version = snapshot.version.checked_add(1).ok_or_else(|| {
                Error::InferenceError("managed KV table version overflow during reset".into())
            })?;
            staged.push((
                group.arena,
                coordinator
                    .stage_table_reset(session, group.domain, next_version)
                    .map_err(coordinator_error)?,
            ));
        }

        // Every fallible ownership/version check completed above. EngineCore
        // serializes manager mutation, so applying the staged resets cannot
        // race another reservation between domains.
        for (arena, reset) in staged {
            state
                .coordinators
                .get_mut(&arena)
                .expect("staged reset arena remains registered")
                .apply_staged_table_reset(reset);
        }
        state.session_generations.insert(session.clone(), next);
        Ok(next)
    }

    /// Drain and retire every arena belonging to one exact loaded-model
    /// generation. The model-scoped physical lease is retained until no
    /// session, row transaction, device fence, or external runtime handle can
    /// still reference the backing storage.
    pub(crate) fn prepare_unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        self.prepare_unload_model_with_runtime_owners(model_instance, 1)
    }

    pub(crate) fn prepare_unload_model_with_runtime_owners(
        &mut self,
        model_instance: ModelInstanceId,
        expected_runtime_owners: usize,
    ) -> Result<bool> {
        if expected_runtime_owners == 0 {
            return Err(Error::InvalidInput(
                "managed KV unload expected zero runtime owners".into(),
            ));
        }
        let Some(state) = self.models.get_mut(&model_instance) else {
            return Ok(false);
        };
        state.closing = true;
        if !state.registered_sessions.is_empty() {
            return Err(Error::InferenceError(format!(
                "managed KV model {} still has registered sessions",
                model_instance.get()
            )));
        }
        if !state.capacity_claims.is_empty() {
            return Err(Error::InferenceError(format!(
                "managed KV model {} still has logical capacity claims",
                model_instance.get()
            )));
        }
        if Arc::strong_count(&state.runtime) != expected_runtime_owners {
            return Err(Error::InferenceError(format!(
                "managed KV model {} has {} runtime owners, expected {expected_runtime_owners}",
                model_instance.get(),
                Arc::strong_count(&state.runtime),
            )));
        }
        // Physical caches clone the arena handle, not the runtime. A clone
        // that outlived its step would keep device backing alive after the
        // lease is released, so the runtime must be the arena's only owner.
        if let Some((arena, owners)) = state
            .runtime
            .arenas
            .iter()
            .map(|(id, arena)| (id, Arc::strong_count(arena)))
            .find(|(_, owners)| *owners != 1)
        {
            return Err(Error::InferenceError(format!(
                "managed KV model {} arena generation {} has {owners} owners, expected 1",
                model_instance.get(),
                arena.generation,
            )));
        }
        for group in &state.runtime.plan.groups {
            loop {
                let evicted = state
                    .prefix_indexes
                    .get_mut(&group.arena)
                    .expect("resolved arena has a prefix index")
                    .evict_lru(
                        state
                            .coordinators
                            .get_mut(&group.arena)
                            .expect("resolved arena has a coordinator"),
                    )
                    .map_err(prefix_error)?;
                if evicted.is_empty() {
                    break;
                }
            }
        }
        for coordinator in state.coordinators.values() {
            let stats = coordinator.stats();
            if stats.allocated_pages != 0
                || stats.table_refs != 0
                || stats.prefix_refs != 0
                || stats.execution_pins != 0
                || stats.transfer_pins != 0
                || stats.reservations != 0
                || stats.active_transactions != 0
            {
                return Err(Error::InferenceError(format!(
                    "managed KV model {} still has live page ownership or transactions",
                    model_instance.get()
                )));
            }
        }
        for arena in state.runtime.arenas.values() {
            arena.drain()?;
        }
        // Committed prefix snapshots are the last tensor state the model
        // holds (no session is registered): free them with their claim.
        clear_tensor_snapshots(state)?;
        if let Some(lease) = state.resource_lease.as_ref() {
            lease.prepare_materialized_release(ResourceVector::zero())?;
        }
        Ok(true)
    }

    pub(crate) fn commit_prepared_unload_model(&mut self, model_instance: ModelInstanceId) -> bool {
        self.models.remove(&model_instance).is_some()
    }

    pub(crate) fn unload_model(&mut self, model_instance: ModelInstanceId) -> Result<bool> {
        if !self.prepare_unload_model(model_instance)? {
            return Ok(false);
        }
        Ok(self.commit_prepared_unload_model(model_instance))
    }

    #[cfg(test)]
    pub(crate) fn snapshot(
        &self,
        model_instance: ModelInstanceId,
        session: &SessionKey,
        domain: CacheDomainId,
    ) -> Option<KvSnapshot> {
        let state = self.models.get(&model_instance)?;
        state
            .coordinators
            .values()
            .find_map(|coordinator| coordinator.snapshot(session, domain).ok())
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn add_coordinator_stats(
    totals: &mut ManagedKvCoordinatorSnapshot,
    arena: &ManagedKvCoordinatorSnapshot,
) {
    totals.capacity_pages = totals.capacity_pages.saturating_add(arena.capacity_pages);
    totals.allocated_pages = totals.allocated_pages.saturating_add(arena.allocated_pages);
    totals.free_pages = totals.free_pages.saturating_add(arena.free_pages);
    totals.admission_claimed_pages = totals
        .admission_claimed_pages
        .saturating_add(arena.admission_claimed_pages);
    totals.admission_available_pages = totals
        .admission_available_pages
        .saturating_add(arena.admission_available_pages);
    totals.admission_claims = totals
        .admission_claims
        .saturating_add(arena.admission_claims);
    totals.table_refs = totals.table_refs.saturating_add(arena.table_refs);
    totals.prefix_refs = totals.prefix_refs.saturating_add(arena.prefix_refs);
    totals.execution_pins = totals.execution_pins.saturating_add(arena.execution_pins);
    totals.transfer_pins = totals.transfer_pins.saturating_add(arena.transfer_pins);
    totals.reservations = totals.reservations.saturating_add(arena.reservations);
    totals.active_transactions = totals
        .active_transactions
        .saturating_add(arena.active_transactions);
}

pub(super) fn managed_backend_runtime(
    backend: BackendKind,
    device: &Device,
) -> (Option<Arc<dyn KvBackendRuntime>>, Option<String>) {
    let wrong_device = || {
        (
            None,
            Some(format!(
                "managed {backend:?} KV cannot bind worker device {:?}",
                device.location()
            )),
        )
    };
    match backend {
        BackendKind::Cpu => {
            if !device.is_cpu() {
                return wrong_device();
            }
            (Some(Arc::new(CpuKvBackendRuntime)), None)
        }
        BackendKind::Metal => {
            if !device.is_metal() {
                return wrong_device();
            }
            #[cfg(feature = "metal")]
            {
                match MetalKvBackendRuntime::new(device.clone()) {
                    Ok(runtime) => (Some(Arc::new(runtime)), None),
                    Err(error) => (None, Some(error.to_string())),
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                (
                    None,
                    Some(
                        "managed Metal KV requires the metal feature and direct paged attention"
                            .to_string(),
                    ),
                )
            }
        }
        BackendKind::Cuda => {
            if !device.is_cuda() {
                return wrong_device();
            }
            #[cfg(feature = "cuda")]
            {
                match CudaKvBackendRuntime::new(device.clone()) {
                    Ok(runtime) => (Some(Arc::new(runtime)), None),
                    Err(error) => (None, Some(error.to_string())),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                (
                    None,
                    Some(
                        "managed CUDA KV requires the cuda feature and direct paged attention"
                            .to_string(),
                    ),
                )
            }
        }
    }
}

pub(super) fn managed_device_ordinal(device: &Device) -> Option<u32> {
    match device.location() {
        DeviceLocation::Cpu => None,
        DeviceLocation::Cuda { gpu_id } => u32::try_from(gpu_id).ok(),
        // Candle reports Metal's registry id rather than the selector ordinal.
        // Fold the exact device identity into the plan's compact device tag;
        // the runtime itself retains the exact Candle device handle.
        DeviceLocation::Metal { gpu_id } => {
            let id = gpu_id as u64;
            Some((id ^ (id >> 32)) as u32)
        }
    }
}

// Kept crate-visible so model-owned contract tests can run the exact shared
// capacity planner without allocating a backend arena.
#[cfg(test)]
pub(crate) fn plan_managed_state_capacity(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    request: ManagedStateCapacityRequest,
) -> Result<(StateRuntimeAllocationPlan, Option<TensorStateCapacity>)> {
    plan_managed_state_capacity_with_policy(state_plan, model_instance, request, false, 0)
}

/// `prefix_snapshot_rows` adds committed prefix snapshots to the lazily
/// materialized tensor envelope (see
/// [`ManagedKvCacheManager::prefix_snapshot_rows_for`]), so the allocation
/// plan, the lease authorization and the CUDA fit all price them.
fn plan_managed_state_capacity_with_policy(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    request: ManagedStateCapacityRequest,
    materialize_cuda_paged_at_load: bool,
    prefix_snapshot_rows: u32,
) -> Result<(StateRuntimeAllocationPlan, Option<TensorStateCapacity>)> {
    if request.total_paged_pages == 0
        || request.retained_sequence_rows == 0
        || request.staged_transaction_rows == 0
    {
        return Err(Error::InvalidInput(
            "managed state capacity requires non-zero page, sequence, and transaction limits"
                .into(),
        ));
    }
    if request.staged_transaction_rows > request.retained_sequence_rows {
        return Err(Error::InvalidInput(
            "managed state transaction rows cannot exceed retained sequence rows".into(),
        ));
    }
    if state_plan.paged_attention.is_empty() {
        return Err(Error::InvalidInput(
            "managed state capacity requires a paged-attention anchor".into(),
        ));
    }
    let page_sizes = state_plan
        .paged_attention
        .iter()
        .map(|group| group.page_tokens)
        .collect::<Vec<_>>();
    if usize::try_from(request.total_paged_pages).unwrap_or(usize::MAX) < page_sizes.len() {
        return Err(Error::Backpressure(
            "managed KV page budget cannot back every paged state group".into(),
        ));
    }
    let upper = u64::from(request.total_paged_pages)
        .checked_mul(u64::from(
            *page_sizes.iter().max().expect("non-empty page sizes"),
        ))
        .ok_or_else(|| Error::InvalidInput("managed KV token reach overflow".into()))?;
    let required_pages = |tokens: u64| -> Result<u64> {
        page_sizes.iter().try_fold(0_u64, |total, page_tokens| {
            let pages = tokens.div_ceil(u64::from(*page_tokens));
            total
                .checked_add(pages)
                .ok_or_else(|| Error::InvalidInput("managed KV page demand overflow".into()))
        })
    };
    let token_reach = match request.logical_token_reach {
        Some(tokens) if tokens > 0 => tokens,
        Some(_) => {
            return Err(Error::InvalidInput(
                "managed state logical token reach must be non-zero".into(),
            ));
        }
        None => {
            let (mut low, mut high) = (1_u64, upper);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if required_pages(middle)? <= u64::from(request.total_paged_pages) {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        }
    };
    let mut groups =
        Vec::with_capacity(state_plan.paged_attention.len() + state_plan.non_paged.len());
    let mut paged_sequence_capacity = u32::MAX;
    for group in &state_plan.paged_attention {
        let blocks = u32::try_from(token_reach.div_ceil(u64::from(group.page_tokens)))
            .map_err(|_| Error::InvalidInput("managed KV group capacity exceeds u32".into()))?;
        // One retained sequence needs at least one page in every paged group,
        // but a page is not a sequence slot. In particular, the thousands of
        // pages needed for one long-context request must never multiply the
        // per-sequence recurrent/convolution state authorization.
        paged_sequence_capacity = paged_sequence_capacity.min(blocks);
        let strategy = managed_paged_capacity_strategy(
            state_plan.backend,
            blocks,
            materialize_cuda_paged_at_load,
        );
        groups.push(GroupCapacityRequest {
            group: group.group,
            domain: group.domain,
            strategy,
        });
    }
    let sequence_capacity = request.retained_sequence_rows.min(paged_sequence_capacity);
    let transaction_capacity = request.staged_transaction_rows.min(sequence_capacity);
    let snapshot_capacity = if state_plan.non_paged.is_empty() {
        0
    } else {
        prefix_snapshot_rows
    };
    let lazy_blocks = sequence_capacity
        .checked_add(transaction_capacity)
        .and_then(|blocks| blocks.checked_add(snapshot_capacity))
        .ok_or_else(|| Error::InvalidInput("managed tensor state capacity overflow".into()))?;
    for domain in &state_plan.non_paged {
        groups.push(GroupCapacityRequest {
            group: domain.group(),
            domain: domain.domain(),
            strategy: CapacityStrategy::BoundedLazy {
                max_blocks: lazy_blocks,
            },
        });
    }
    groups.sort_unstable_by_key(|group| (group.group, group.domain));
    let workspace = WorkspaceContract {
        fixed_bytes: 0,
        dimensions: vec![],
        terms: vec![],
        placement: match state_plan.backend {
            BackendKind::Cpu => WorkspacePlacement::Host,
            BackendKind::Metal | BackendKind::Cuda => WorkspacePlacement::BackendLocal,
        },
        concurrency_slots: transaction_capacity,
    };
    let registry = StateBackendRegistry::new(state_plan.backend, state_plan.device_ordinal)?;
    let allocation_plan = StateRuntimeAllocationPlan::build_exact(
        state_plan,
        model_instance,
        groups,
        workspace,
        &registry,
    )?;
    let tensor_capacity = (!state_plan.non_paged.is_empty())
        .then(|| {
            TensorStateCapacity::for_plan(state_plan, sequence_capacity, transaction_capacity)?
                .with_snapshot_capacity(snapshot_capacity)
        })
        .transpose()?;
    if let Some(capacity) = tensor_capacity {
        let planned_authorization = capacity
            .per_sequence_bytes()
            .checked_mul(u64::from(lazy_blocks))
            .ok_or_else(|| Error::InvalidInput("tensor authorization overflow".into()))?;
        if capacity.authorized_bytes() != planned_authorization {
            return Err(Error::InferenceError(
                "tensor arena authorization diverges from the allocation plan".into(),
            ));
        }
    }
    Ok((allocation_plan, tensor_capacity))
}

fn managed_paged_capacity_strategy(
    backend: BackendKind,
    blocks: u32,
    materialize_cuda_paged_at_load: bool,
) -> CapacityStrategy {
    if backend == BackendKind::Cuda && blocks > 64 && !materialize_cuda_paged_at_load {
        let growth = cuda_paged_growth_geometry(blocks);
        CapacityStrategy::AdmissionGrowable {
            initial_blocks: growth.initial_pages,
            growth_quantum: growth.growth_quantum_pages,
            max_blocks: blocks,
        }
    } else {
        CapacityStrategy::Fixed { blocks }
    }
}

/// Share of a KV arena, allocated in full at load by the CUDA resident fit,
/// that an automatic prefix page budget may hold: one page in this many.
const AUTOMATIC_PREFIX_ARENA_SHARE: usize = 4;

/// Page budget of one arena's committed-prefix index.
///
/// `budget` is the prefix page budget the cache policy resolved: the
/// operator's value, or [`crate::config::DEFAULT_MAX_PREFIX_CACHE_PAGES`]
/// when `automatic`, and zero when prefix reuse is disabled. An operator's
/// value applies as set, and so does an automatic budget on CPU, on Metal and
/// on CUDA arenas that grow on demand. On a CUDA arena allocated in full at
/// load (`materialized_at_load`), an automatic budget rises to a quarter of
/// the arena. Those pages occupy device memory whether or not the index holds
/// them, and a request that needs pages held only by the index evicts them,
/// least recently used first (`evict_lru_excluding` in `prepare_inner`), so
/// the larger budget only keeps more chat history reusable: a quarter of
/// Qwen3.8's ~9,700-page arena on an H100 is ~2,400 pages, over 150k
/// tokens. A budget never exceeds its arena.
fn prefix_index_pages(
    budget: usize,
    automatic: bool,
    backend: BackendKind,
    materialized_at_load: bool,
    arena_pages: u32,
) -> usize {
    let arena_pages = usize::try_from(arena_pages).unwrap_or(usize::MAX);
    let scales = automatic && backend == BackendKind::Cuda && materialized_at_load;
    let budget = if budget > 0 && scales {
        budget.max(arena_pages / AUTOMATIC_PREFIX_ARENA_SHARE)
    } else {
        budget
    };
    budget.min(arena_pages)
}

/// Maximum simultaneously live page rows for the CUDA arena's current
/// doubling/quantized contiguous replacement schedule.
fn cuda_contiguous_replacement_peak_pages(maximum_pages: u32) -> Result<u64> {
    if maximum_pages == 0 {
        return Err(Error::InvalidInput(
            "CUDA KV replacement peak requires non-zero pages".into(),
        ));
    }
    let geometry = cuda_paged_growth_geometry(maximum_pages);
    let mut current = geometry.initial_pages;
    let mut peak = u64::from(current);
    while current < maximum_pages {
        let quantum = geometry.growth_quantum_pages;
        let rounded_target = current.saturating_add(quantum).min(maximum_pages);
        let amortized_addition = current
            .max(quantum)
            .div_ceil(quantum)
            .saturating_mul(quantum);
        let doubled = current
            .saturating_add(amortized_addition)
            .min(maximum_pages);
        let target = rounded_target.max(doubled).min(maximum_pages);
        peak = peak.max(u64::from(current) + u64::from(target));
        if target <= current {
            return Err(Error::InferenceError(
                "CUDA KV replacement growth made no progress".into(),
            ));
        }
        current = target;
    }
    Ok(peak)
}

fn cuda_largest_contiguous_replacement_extra(
    paged: impl IntoIterator<Item = (u32, u64)>,
) -> Result<u64> {
    paged
        .into_iter()
        .try_fold(0_u64, |largest_extra, (maximum_pages, bytes_per_page)| {
            let peak_pages = cuda_contiguous_replacement_peak_pages(maximum_pages)?;
            let extra_pages = peak_pages
                .checked_sub(u64::from(maximum_pages))
                .ok_or_else(|| {
                    Error::InferenceError("CUDA KV replacement peak underflow".into())
                })?;
            let extra_bytes = bytes_per_page.checked_mul(extra_pages).ok_or_else(|| {
                Error::ModelLoadError("CUDA KV replacement byte plan overflow".into())
            })?;
            Ok(largest_extra.max(extra_bytes))
        })
}

fn cuda_managed_state_peak_authorization(
    steady_resources: ResourceVector,
    groups: &[crate::kv::ResolvedKvGroup],
) -> Result<ResourceVector> {
    let ResourceAmount::Known(steady_device_bytes) = steady_resources.device_bytes else {
        return Err(Error::ModelLoadError(
            "CUDA managed state requires known steady device bytes".into(),
        ));
    };
    let replacement_extra =
        cuda_largest_contiguous_replacement_extra(groups.iter().filter_map(|group| {
            match group.capacity_strategy {
                CapacityStrategy::AdmissionGrowable { .. } => {
                    Some((group.capacity_pages, group.bytes_per_page))
                }
                CapacityStrategy::Fixed { .. }
                | CapacityStrategy::BoundedLazy { .. }
                | CapacityStrategy::Reserved { .. } => None,
            }
        }))?;
    Ok(ResourceVector {
        device_bytes: ResourceAmount::Known(
            steady_device_bytes
                .checked_add(replacement_extra)
                .ok_or_else(|| Error::ModelLoadError("CUDA state peak overflow".into()))?,
        ),
        ..steady_resources
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaContiguousPagedGeometry {
    page_tokens: u32,
    bytes_per_page: u64,
}

/// Operator guidance shared by the pre-load and resident CUDA state checks.
pub(crate) const CUDA_EXPLICIT_ROWS_GUIDANCE: &str = "explicit rows are a hard contract, so lower IZWI_MAX_RETAINED_SEQUENCES and IZWI_MAX_STAGED_TRANSACTIONS until (retained + staged) x per_sequence_state_bytes fits, or unset both to fit rows to the model";
pub(crate) const CUDA_AUTOMATIC_ROWS_GUIDANCE: &str = "rows were automatic and cannot go below the staged width, so free device memory (unload other models or use a larger GPU) or lower IZWI_MAX_STAGED_TRANSACTIONS (keeping IZWI_MAX_RETAINED_SEQUENCES at least as large)";

/// Row limits for `fit_cuda_resident_logical_token_reach`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CudaResidentRowRequest {
    pub(crate) retained_sequence_rows: u32,
    pub(crate) staged_transaction_rows: u32,
    /// `Some(share)` when retained rows are automatic: their recurrent state
    /// may take at most this share of the post-weight budget.
    pub(crate) automatic_state_budget_fraction: Option<f64>,
}

/// Rows and logical reach chosen by the CUDA resident fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaResidentFit {
    pub(crate) logical_token_reach: u64,
    pub(crate) retained_sequence_rows: u32,
    pub(crate) staged_transaction_rows: u32,
    /// Committed prefix snapshots reserved beside the rows.
    pub(crate) prefix_snapshot_rows: u32,
    /// Recurrent tensor state of one row; zero for paged-only plans.
    pub(crate) per_sequence_state_bytes: u64,
    /// Recurrent tensor state reserved for every retained and staged row and
    /// every prefix snapshot.
    pub(crate) resident_state_bytes: u64,
}

/// Choose retained rows and the logical reach for one resolved plan. Only
/// the plan's geometry is read, so CPU tests can drive the CUDA policy.
///
/// `prefix_snapshot_rows` committed prefix snapshots are fixed state: they
/// count against the automatic rows' state share (so automatic retained rows
/// shrink by about that many, and KV is unchanged), and against the whole
/// budget beside explicit rows.
#[allow(clippy::too_many_arguments)]
fn fit_resident_rows_and_reach(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    maximum_tokens_for_rows: impl Fn(u32) -> Result<u64>,
    headroom_bytes: u64,
    safety_reserve_bytes: u64,
    page_tokens_hint: u32,
    rows: CudaResidentRowRequest,
    prefix_snapshot_rows: u32,
) -> Result<CudaResidentFit> {
    let budget_bytes = headroom_bytes.saturating_sub(safety_reserve_bytes);
    let geometry = |retained_sequence_rows: u32| {
        cuda_resident_geometry(
            state_plan,
            model_instance,
            maximum_tokens_for_rows(retained_sequence_rows)?,
            page_tokens_hint,
            retained_sequence_rows,
            rows.staged_transaction_rows.min(retained_sequence_rows),
            prefix_snapshot_rows,
        )
    };
    let chosen = match rows.automatic_state_budget_fraction {
        None => {
            let explicit = geometry(rows.retained_sequence_rows)?;
            if explicit.minimum_bytes > budget_bytes {
                tracing::warn!(
                    "CUDA managed context cannot fit the model-authored minimum: {}",
                    explicit.describe(headroom_bytes, safety_reserve_bytes)
                );
                return Err(Error::ModelLoadError(format!(
                    "{}; {CUDA_EXPLICIT_ROWS_GUIDANCE}",
                    crate::engine::resources::capacity_shortfall_reason(
                        "CUDA managed context cannot fit the model-authored minimum",
                        explicit.minimum_bytes,
                        budget_bytes,
                    )
                )));
            }
            explicit
        }
        Some(fraction) => {
            let floor = geometry(
                rows.staged_transaction_rows
                    .min(rows.retained_sequence_rows),
            )?;
            if floor.minimum_bytes > budget_bytes {
                tracing::warn!(
                    "CUDA managed context cannot fit the model-authored minimum at the decode width: {}",
                    floor.describe(headroom_bytes, safety_reserve_bytes)
                );
                return Err(Error::ModelLoadError(format!(
                    "{}; {CUDA_AUTOMATIC_ROWS_GUIDANCE}",
                    crate::engine::resources::capacity_shortfall_reason(
                        "CUDA managed context cannot fit the model-authored minimum at the decode width",
                        floor.minimum_bytes,
                        budget_bytes,
                    )
                )));
            }
            let state_cap = (budget_bytes as f64 * fraction) as u64;
            let fits = |candidate: &CudaResidentGeometry| {
                candidate.resident_state_bytes <= state_cap
                    && candidate.minimum_bytes <= budget_bytes
            };
            // The floor is the decode width: it may exceed the share but
            // must stay usable. Above it, take the most rows that fit.
            let mut chosen = floor;
            if fits(&floor) {
                let (mut low, mut high) =
                    (floor.retained_sequence_rows, rows.retained_sequence_rows);
                while low < high {
                    let middle = low + (high - low).div_ceil(2);
                    let candidate = geometry(middle)?;
                    if fits(&candidate) {
                        low = middle;
                        chosen = candidate;
                    } else {
                        high = middle - 1;
                    }
                }
            }
            chosen
        }
    };
    let paged = state_plan
        .paged_attention
        .iter()
        .map(|paged| CudaContiguousPagedGeometry {
            page_tokens: paged.page_tokens,
            bytes_per_page: paged.bytes_per_page,
        })
        .collect::<Vec<_>>();
    Ok(CudaResidentFit {
        logical_token_reach: fit_cuda_resident_token_reach(
            chosen.maximum_tokens,
            page_tokens_hint,
            &paged,
            chosen.resident_state_bytes,
            budget_bytes,
        )?,
        retained_sequence_rows: chosen.retained_sequence_rows,
        staged_transaction_rows: chosen.staged_transaction_rows,
        prefix_snapshot_rows: chosen.prefix_snapshot_rows,
        per_sequence_state_bytes: chosen.per_sequence_state_bytes,
        resident_state_bytes: chosen.resident_state_bytes,
    })
}

/// Fixed state and minimum resident footprint of one candidate row count.
#[derive(Debug, Clone, Copy)]
struct CudaResidentGeometry {
    retained_sequence_rows: u32,
    staged_transaction_rows: u32,
    prefix_snapshot_rows: u32,
    maximum_tokens: u64,
    per_sequence_state_bytes: u64,
    resident_state_bytes: u64,
    minimum_tokens: u64,
    minimum_bytes: u64,
}

impl CudaResidentGeometry {
    fn describe(&self, headroom_bytes: u64, safety_reserve_bytes: u64) -> String {
        format!(
            "retained_sequences={}, staged_transactions={}, prefix_snapshots={}, per_sequence_state_bytes={}, resident_state_bytes={}, minimum_tokens={}, minimum_bytes={}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}",
            self.retained_sequence_rows,
            self.staged_transaction_rows,
            self.prefix_snapshot_rows,
            self.per_sequence_state_bytes,
            self.resident_state_bytes,
            self.minimum_tokens,
            self.minimum_bytes,
        )
    }
}

/// Price one row count: the fixed (non-paged) state the allocation plan
/// authorizes for retained plus staged rows, and that state plus one page per
/// paged domain. Only plan geometry is read, so a CPU-negotiated plan of the
/// same contract prices the same bytes in tests.
fn cuda_resident_geometry(
    state_plan: &ResolvedStatePlan,
    model_instance: ModelInstanceId,
    maximum_tokens: u64,
    page_tokens: u32,
    retained_sequence_rows: u32,
    staged_transaction_rows: u32,
    prefix_snapshot_rows: u32,
) -> Result<CudaResidentGeometry> {
    let (maximum_allocation, tensor_capacity) = plan_managed_state_capacity_with_policy(
        state_plan,
        model_instance,
        ManagedStateCapacityRequest {
            total_paged_pages: u32::MAX,
            logical_token_reach: Some(maximum_tokens),
            retained_sequence_rows,
            staged_transaction_rows,
        },
        false,
        prefix_snapshot_rows,
    )?;
    let paged = state_plan
        .paged_attention
        .iter()
        .map(|paged| CudaContiguousPagedGeometry {
            page_tokens: paged.page_tokens,
            bytes_per_page: paged.bytes_per_page,
        })
        .collect::<Vec<_>>();
    let steady_resources = managed_state_resources(
        state_plan.backend,
        maximum_allocation.maximum_resources(state_plan)?,
    )?;
    let steady_bytes = match state_plan.backend {
        BackendKind::Cuda => known_resource_bytes(steady_resources.device_bytes, "device"),
        BackendKind::Metal => known_resource_bytes(steady_resources.unified_bytes, "unified"),
        BackendKind::Cpu => known_resource_bytes(steady_resources.host_bytes, "host"),
    }?;
    let maximum_paged_bytes =
        state_plan
            .paged_attention
            .iter()
            .try_fold(0_u64, |total, paged| {
                let maximum_blocks = maximum_allocation
                    .group_capacity(paged.group, paged.domain)?
                    .strategy
                    .maximum_blocks();
                total
                    .checked_add(
                        paged
                            .bytes_per_page
                            .checked_mul(u64::from(maximum_blocks))
                            .ok_or_else(|| {
                                Error::ModelLoadError("CUDA KV byte plan overflow".into())
                            })?,
                    )
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))
            })?;
    let resident_state_bytes = steady_bytes
        .checked_sub(maximum_paged_bytes)
        .ok_or_else(|| Error::ModelLoadError("CUDA fixed state byte plan underflow".into()))?;
    let minimum_tokens = u64::from(page_tokens).min(maximum_tokens);
    Ok(CudaResidentGeometry {
        retained_sequence_rows,
        staged_transaction_rows,
        prefix_snapshot_rows: tensor_capacity.map_or(0, |capacity| capacity.snapshot_capacity()),
        maximum_tokens,
        per_sequence_state_bytes: tensor_capacity
            .map_or(0, |capacity| capacity.per_sequence_bytes()),
        resident_state_bytes,
        minimum_tokens,
        minimum_bytes: cuda_resident_required_bytes(minimum_tokens, &paged, resident_state_bytes)?,
    })
}

/// Price the worst admission-barrier replacement across independently grown
/// paged arenas. Every arena may already be at its final steady backing when
/// another arena grows, but only the arena currently being replaced retains an
/// additional old backing alongside its target allocation.
#[cfg(test)]
fn cuda_contiguous_replacement_required_bytes(
    tokens: u64,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
) -> Result<u64> {
    if paged.is_empty() {
        return Err(Error::InvalidInput(
            "CUDA contiguous context fitting requires paged state".into(),
        ));
    }
    let (steady_paged_bytes, largest_replacement_extra) =
        paged
            .iter()
            .try_fold((0_u64, 0_u64), |(steady_total, largest_extra), geometry| {
                if geometry.page_tokens == 0 || geometry.bytes_per_page == 0 {
                    return Err(Error::InvalidInput(
                        "CUDA contiguous context fitting requires non-zero paged geometry".into(),
                    ));
                }
                let pages = u32::try_from(tokens.div_ceil(u64::from(geometry.page_tokens)))
                    .map_err(|_| Error::ModelLoadError("CUDA KV page demand exceeds u32".into()))?;
                let peak_pages = cuda_contiguous_replacement_peak_pages(pages)?;
                let steady_bytes = geometry
                    .bytes_per_page
                    .checked_mul(u64::from(pages))
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))?;
                let peak_bytes = geometry
                    .bytes_per_page
                    .checked_mul(peak_pages)
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))?;
                let replacement_extra = peak_bytes.checked_sub(steady_bytes).ok_or_else(|| {
                    Error::ModelLoadError("CUDA KV replacement peak underflow".into())
                })?;
                Ok((
                    steady_total.checked_add(steady_bytes).ok_or_else(|| {
                        Error::ModelLoadError("CUDA state peak byte plan overflow".into())
                    })?,
                    largest_extra.max(replacement_extra),
                ))
            })?;
    fixed_state_bytes
        .checked_add(steady_paged_bytes)
        .and_then(|bytes| bytes.checked_add(largest_replacement_extra))
        .ok_or_else(|| Error::ModelLoadError("CUDA state peak byte plan overflow".into()))
}

fn cuda_resident_required_bytes(
    tokens: u64,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
) -> Result<u64> {
    if paged.is_empty() {
        return Err(Error::InvalidInput(
            "CUDA resident context fitting requires paged state".into(),
        ));
    }
    paged.iter().try_fold(fixed_state_bytes, |total, geometry| {
        if geometry.page_tokens == 0 || geometry.bytes_per_page == 0 {
            return Err(Error::InvalidInput(
                "CUDA resident context fitting requires non-zero paged geometry".into(),
            ));
        }
        let pages = tokens.div_ceil(u64::from(geometry.page_tokens));
        total
            .checked_add(
                geometry
                    .bytes_per_page
                    .checked_mul(pages)
                    .ok_or_else(|| Error::ModelLoadError("CUDA KV byte plan overflow".into()))?,
            )
            .ok_or_else(|| Error::ModelLoadError("CUDA state byte plan overflow".into()))
    })
}

fn fit_cuda_resident_token_reach(
    maximum_tokens: u64,
    minimum_page_tokens: u32,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
    budget_bytes: u64,
) -> Result<u64> {
    let minimum_tokens = u64::from(minimum_page_tokens).min(maximum_tokens);
    if cuda_resident_required_bytes(maximum_tokens, paged, fixed_state_bytes)? <= budget_bytes {
        return Ok(maximum_tokens);
    }
    let (mut low, mut high) = (minimum_tokens, maximum_tokens);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if cuda_resident_required_bytes(middle, paged, fixed_state_bytes)? <= budget_bytes {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

#[cfg(test)]
fn fit_cuda_contiguous_token_reach(
    maximum_tokens: u64,
    minimum_page_tokens: u32,
    paged: &[CudaContiguousPagedGeometry],
    fixed_state_bytes: u64,
    budget_bytes: u64,
) -> Result<u64> {
    let minimum_tokens = u64::from(minimum_page_tokens).min(maximum_tokens);
    if cuda_contiguous_replacement_required_bytes(maximum_tokens, paged, fixed_state_bytes)?
        <= budget_bytes
    {
        return Ok(maximum_tokens);
    }
    let (mut low, mut high) = (minimum_tokens, maximum_tokens);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if cuda_contiguous_replacement_required_bytes(middle, paged, fixed_state_bytes)?
            <= budget_bytes
        {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

fn known_resource_bytes(amount: ResourceAmount, domain: &str) -> Result<u64> {
    match amount {
        ResourceAmount::Known(bytes) => Ok(bytes),
        ResourceAmount::Unknown => Err(Error::ModelLoadError(format!(
            "portable context planning has unknown {domain} capacity"
        ))),
    }
}

fn managed_state_resources(
    backend: BackendKind,
    state: StateResourceVector,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    let host = state
        .host_bytes
        .checked_add(state.pinned_bytes)
        .and_then(|bytes| bytes.checked_add(state.metadata_bytes))
        .ok_or_else(|| Error::Overloaded("managed state host byte total overflow".into()))?;
    match backend {
        BackendKind::Cpu => {
            let total = host
                .checked_add(state.device_bytes)
                .ok_or_else(|| Error::Overloaded("managed CPU state total overflow".into()))?;
            resources.host_bytes = ResourceAmount::Known(total);
        }
        BackendKind::Metal => {
            let total = host
                .checked_add(state.device_bytes)
                .ok_or_else(|| Error::Overloaded("managed Metal state total overflow".into()))?;
            resources.unified_bytes = ResourceAmount::Known(total);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host);
            resources.device_bytes = ResourceAmount::Known(state.device_bytes);
        }
    }
    Ok(resources)
}

/// Tensor-state bytes in the backend's lease domain (device on CUDA, unified
/// on Metal, host on CPU), as `managed_state_resources` prices the envelope.
fn tensor_state_resources(backend: BackendKind, bytes: u64) -> Result<ResourceVector> {
    managed_state_resources(
        backend,
        StateResourceVector {
            device_bytes: bytes,
            ..StateResourceVector::default()
        },
    )
}

/// Move the tensor arena's committed backing between the model lease's
/// pending and materialized claims. The tensor envelope is authorized at load
/// but allocated lazily: without this, every committed row stays pending
/// while also shrinking live free memory, so guarded admission charges it
/// twice. Growth is recorded after the state exists (commits publish tensors
/// the model already allocated). A shrink restores the pending claim, so the
/// caller must drop the [`RetiredTensorState`] only after this returns.
///
/// Every caller has already applied its arena transition, so a failure here
/// is logged rather than returned: a stale record keeps the previous claim.
fn reconcile_tensor_materialization(state: &mut ManagedKvModelState) {
    if state.tensor_rows_request_priced {
        return;
    }
    let Some(arena) = state.runtime.tensor_state() else {
        return;
    };
    let result = (|| -> Result<Option<u64>> {
        let bytes = arena
            .committed_backing_bytes()?
            .min(arena.capacity().authorized_bytes());
        if bytes == state.tensor_materialized_bytes {
            return Ok(None);
        }
        if let Some(lease) = state.resource_lease.as_ref() {
            let materialized = state
                .materialized_resources
                .checked_add(tensor_state_resources(state.runtime.plan.backend, bytes)?)?;
            if bytes > state.tensor_materialized_bytes {
                lease.record_materialized_usage(materialized)?;
            } else {
                lease.prepare_materialized_release(materialized)?;
            }
        }
        Ok(Some(bytes))
    })();
    match result {
        Ok(Some(bytes)) => state.tensor_materialized_bytes = bytes,
        Ok(None) => {}
        Err(error) => tracing::error!(
            error = %error,
            recorded_bytes = state.tensor_materialized_bytes,
            "managed tensor-state materialization could not be recorded on the model lease"
        ),
    }
}

/// Release one tensor sequence, restoring its pending claim before its state
/// can be freed.
fn release_tensor_sequence(
    state: &mut ManagedKvModelState,
    arena: &TensorStateArena,
    sequence: PhysicalStateSequenceId,
) -> Result<()> {
    let retired = arena.release(sequence)?;
    reconcile_tensor_materialization(state);
    drop(retired);
    Ok(())
}

fn reserve_managed_arena(
    authority: &Arc<ResourceAuthority>,
    model_instance: ModelInstanceId,
    backend: BackendKind,
    resources: ResourceVector,
) -> Result<ResourceLease> {
    let owner = ReservationOwner::new(
        ReservationClass::Model,
        format!("managed-kv:{}:{backend:?}", model_instance.get()),
    );
    authority.reserve(owner, resources)
}

fn ensure_session_tables(state: &mut ManagedKvModelState, session: &SessionKey) -> Result<()> {
    if !state.registered_sessions.insert(session.clone()) {
        return Ok(());
    }
    let generation = ManagedSessionGeneration::INITIAL;
    state
        .session_generations
        .insert(session.clone(), generation);
    let mut registered = Vec::new();
    for group in &state.runtime.plan.groups {
        let coordinator = state
            .coordinators
            .get_mut(&group.arena)
            .expect("resolved arena has a coordinator");
        if let Err(error) = coordinator.register_table(session.clone(), group.domain) {
            for (arena, domain) in registered {
                let _ = state
                    .coordinators
                    .get_mut(&arena)
                    .expect("registered arena exists")
                    .release_table(session, domain);
            }
            state.registered_sessions.remove(session);
            state.session_generations.remove(session);
            return Err(coordinator_error(error));
        }
        registered.push((group.arena, group.domain));
    }
    Ok(())
}

fn validate_reservation_session_generation(
    state: &ManagedKvModelState,
    reservation: &ManagedCacheReservation,
) -> Result<()> {
    let current = state
        .session_generations
        .get(&reservation.session)
        .copied()
        .ok_or_else(|| {
            Error::InferenceError(
                "managed KV reservation has no registered session generation".into(),
            )
        })?;
    if reservation.session_generation != current {
        return Err(Error::InferenceError(format!(
            "managed KV reservation session generation {} is stale; current generation is {}",
            reservation.session_generation.get(),
            current.get()
        )));
    }
    Ok(())
}

/// Promise enough logical pages for the request's full prompt plus maximum
/// output before its first CUDA/Metal/CPU dispatch. The promise is deliberately
/// separate from physical page ownership: pages are still materialized and
/// written incrementally, while admission cannot overbook the pool and fail at
/// an arbitrary later decode step.
/// How a row's logical managed-KV admission claim is sized. Claims never
/// exceed arena capacity in total and a step runs only when its own claim
/// covers it, so every policy keeps physical pages within capacity mid-decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ManagedClaimPolicy {
    /// Prompt plus the whole output budget, claimed once at admission.
    Full,
    /// Prompt plus a bounded output quantum, extended as generation proceeds
    /// (vLLM-style on-demand admission). An omitted OpenAI `max_tokens`
    /// becomes "rest of context", so a full claim let one default request
    /// reserve the entire arena and serialize every other request behind it.
    Chunked { quantum_tokens: u32 },
    /// Exactly the next scheduled span, including adapter-projected
    /// speculative spans (the CUDA Qwen3.8 replay-capable route).
    Incremental,
}

/// Claim the prompt and the current step, plus up to `quantum_tokens` of
/// output headroom, never more than the request's prompt-plus-output budget.
/// Only the current step is required: if other claims leave no room for the
/// headroom the claim grows just enough for this step, and if not even that
/// fits the row waits for capacity (`Backpressure`). A claim never shrinks.
fn ensure_chunked_capacity_claim(
    state: &mut ManagedKvModelState,
    session: &SessionKey,
    request: &EngineCoreRequest,
    work: &WorkUnit,
    quantum_tokens: u32,
) -> Result<bool> {
    let WorkUnit::SequenceStep { input, phase, .. } = work else {
        return ensure_capacity_claim(state, session, request);
    };
    let prompt = u64::try_from(request.num_prompt_tokens())
        .map_err(|_| Error::Overloaded("chunked token demand exceeds u64".into()))?;
    let budget = prompt
        .checked_add(u64::try_from(request.params.max_tokens.max(1)).unwrap_or(u64::MAX))
        .ok_or_else(|| Error::Overloaded("chunked token demand overflowed".into()))?;
    let mut demand_by_arena = HashMap::<KvArenaId, (u64, u64)>::new();
    for group in &state.runtime.plan.groups {
        let projected = request.project_paged_state_input(group.domain, *phase, *input)?;
        let needed = prompt.max(
            u64::try_from(projected.end)
                .map_err(|_| Error::Overloaded("chunked token demand exceeds u64".into()))?,
        );
        let desired = budget
            .max(needed)
            .min(needed.saturating_add(u64::from(quantum_tokens)));
        // A sliding window retains at most the window, the span being written
        // and one page of alignment, exactly as the incremental claim prices.
        let retained = |tokens: u64| -> Result<u64> {
            Ok(sliding_window_for_domain(&state.contract, group.domain)?
                .map(|window| {
                    tokens.min(
                        u64::from(window)
                            .saturating_add(projected.len() as u64)
                            .saturating_add(u64::from(group.page_tokens - 1)),
                    )
                })
                .unwrap_or(tokens))
        };
        let page_tokens = u64::from(group.page_tokens);
        let entry = demand_by_arena.entry(group.arena).or_default();
        entry.0 = entry
            .0
            .checked_add(retained(needed)?.div_ceil(page_tokens))
            .ok_or_else(|| Error::Overloaded("chunked arena page demand overflow".into()))?;
        entry.1 = entry
            .1
            .checked_add(retained(desired)?.div_ceil(page_tokens))
            .ok_or_else(|| Error::Overloaded("chunked arena page demand overflow".into()))?;
    }
    let mut claims = Vec::with_capacity(demand_by_arena.len());
    for (arena, (needed_pages, desired_pages)) in demand_by_arena {
        let group = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == arena)
            .ok_or_else(|| Error::InferenceError("managed capacity arena disappeared".into()))?;
        let capacity = u64::from(group.capacity_pages);
        if needed_pages > capacity {
            let token_capacity = capacity.saturating_mul(u64::from(group.page_tokens));
            return Err(Error::Overloaded(format!(
                "request {} needs {needed_pages} pages for its next step, but managed state arena {} can retain at most {token_capacity} tokens ({capacity} pages of {} tokens)",
                request.id,
                group.id.get(),
                group.page_tokens
            )));
        }
        let previous = state
            .capacity_claims
            .get(session)
            .and_then(|claims| claims.iter().find(|(claimed, _)| *claimed == arena))
            .map(|(_, pages)| u64::from(*pages))
            .unwrap_or(0);
        let others = state
            .capacity_claims
            .iter()
            .filter(|(owner, _)| *owner != session)
            .flat_map(|(_, claims)| claims)
            .filter(|(claimed, _)| *claimed == arena)
            .map(|(_, pages)| u64::from(*pages))
            .sum::<u64>();
        let available = capacity.saturating_sub(others);
        let required = previous.max(needed_pages);
        if required > available {
            return Err(Error::Backpressure(format!(
                "managed KV chunked admission is waiting: request {} needs {required} pages, {others} are claimed by other requests, capacity is {capacity}",
                request.id
            )));
        }
        let pages = required.max(desired_pages.min(available));
        claims.push((
            arena,
            u32::try_from(pages)
                .map_err(|_| Error::Overloaded("chunked page claim exceeds u32".into()))?,
        ));
    }
    let newly_installed = !state.capacity_claims.contains_key(session);
    state.capacity_claims.insert(session.clone(), claims);
    Ok(newly_installed)
}

fn ensure_capacity_claim(
    state: &mut ManagedKvModelState,
    session: &SessionKey,
    request: &EngineCoreRequest,
) -> Result<bool> {
    if state.capacity_claims.contains_key(session) {
        return Ok(false);
    }
    let requested_tokens = request
        .num_prompt_tokens()
        .checked_add(request.params.max_tokens.max(1))
        .ok_or_else(|| Error::Overloaded("request token capacity demand overflowed".into()))?;
    let requested_tokens = u64::try_from(requested_tokens)
        .map_err(|_| Error::Overloaded("request token capacity demand exceeds u64".into()))?;

    let mut requested_by_arena = HashMap::<KvArenaId, u64>::new();
    for group in &state.runtime.plan.groups {
        let window_tokens = sliding_window_for_domain(&state.contract, group.domain)?
            .map(u64::from)
            .unwrap_or(requested_tokens);
        let retained_tokens = requested_tokens.min(window_tokens);
        let required_pages = retained_tokens.div_ceil(u64::from(group.page_tokens));
        let requested = requested_by_arena.entry(group.arena).or_default();
        *requested = requested.saturating_add(required_pages);
    }

    for (arena, requested_pages) in &requested_by_arena {
        let capacity_pages = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == *arena)
            .map(|group| u64::from(group.capacity_pages))
            .ok_or_else(|| Error::InferenceError("managed capacity arena disappeared".into()))?;
        if *requested_pages > capacity_pages {
            let group = state
                .runtime
                .plan
                .groups
                .iter()
                .find(|group| group.arena == *arena)
                .expect("capacity arena group exists");
            let token_capacity = capacity_pages.saturating_mul(u64::from(group.page_tokens));
            return Err(Error::Overloaded(format!(
                "request {} needs {requested_tokens} prompt-plus-output tokens, but managed state arena {} can retain at most {token_capacity} tokens ({} pages of {} tokens)",
                request.id,
                group.id.get(),
                capacity_pages,
                group.page_tokens
            )));
        }
        let claimed_pages = state
            .capacity_claims
            .values()
            .flat_map(|claims| claims.iter())
            .filter_map(|(claimed_arena, pages)| {
                (*claimed_arena == *arena).then_some(u64::from(*pages))
            })
            .sum::<u64>();
        if claimed_pages.saturating_add(*requested_pages) > capacity_pages {
            return Err(Error::Backpressure(format!(
                "managed KV full-request admission is waiting: request {} needs {} pages, {} are already claimed, and arena capacity is {} pages",
                request.id, requested_pages, claimed_pages, capacity_pages
            )));
        }
    }

    let claims = requested_by_arena
        .into_iter()
        .map(|(arena, pages)| {
            u32::try_from(pages)
                .map(|pages| (arena, pages))
                .map_err(|_| Error::Overloaded("managed KV page claim exceeds u32".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    state.capacity_claims.insert(session.clone(), claims);
    Ok(true)
}

/// Claim the known prompt and the exact next scheduled progress (including
/// adapter-projected speculative spans), never the potential complete answer.
/// Validate all arenas before publishing any changes to the session claim.
fn ensure_incremental_capacity_claim(
    state: &mut ManagedKvModelState,
    session: &SessionKey,
    request: &EngineCoreRequest,
    work: &WorkUnit,
) -> Result<bool> {
    let WorkUnit::SequenceStep { input, phase, .. } = work else {
        return ensure_capacity_claim(state, session, request);
    };
    let mut requested_by_arena = HashMap::<KvArenaId, u64>::new();
    for group in &state.runtime.plan.groups {
        let projected = request.project_paged_state_input(group.domain, *phase, *input)?;
        let target = u64::try_from(request.num_prompt_tokens().max(projected.end))
            .map_err(|_| Error::Overloaded("incremental token demand exceeds u64".into()))?;
        // Window movement can retain the old window while preparing the next
        // chunk; reserve its append and page alignment as well.
        let retained = sliding_window_for_domain(&state.contract, group.domain)?
            .map(|window| {
                target.min(
                    u64::from(window)
                        .saturating_add(projected.len() as u64)
                        .saturating_add(u64::from(group.page_tokens - 1)),
                )
            })
            .unwrap_or(target);
        let required_pages = retained.div_ceil(u64::from(group.page_tokens));
        let requested = requested_by_arena.entry(group.arena).or_default();
        *requested = requested
            .checked_add(required_pages)
            .ok_or_else(|| Error::Overloaded("incremental arena page demand overflow".into()))?;
    }
    let mut claims = Vec::with_capacity(requested_by_arena.len());
    for (arena, requested_pages) in requested_by_arena {
        let group = state
            .runtime
            .plan
            .groups
            .iter()
            .find(|group| group.arena == arena)
            .ok_or_else(|| Error::InferenceError("managed capacity arena disappeared".into()))?;
        let previous = state
            .capacity_claims
            .get(session)
            .and_then(|claims| claims.iter().find(|(arena, _)| *arena == group.arena))
            .map(|(_, pages)| u64::from(*pages))
            .unwrap_or(0);
        let pages = requested_pages.max(previous);
        let capacity = u64::from(group.capacity_pages);
        if pages > capacity {
            return Err(Error::Overloaded(format!(
                "request {} next managed step needs {pages} pages, arena capacity is {capacity}",
                request.id
            )));
        }
        let others = state
            .capacity_claims
            .iter()
            .filter(|(owner, _)| *owner != session)
            .flat_map(|(_, claims)| claims)
            .filter(|(arena, _)| *arena == group.arena)
            .map(|(_, pages)| u64::from(*pages))
            .sum::<u64>();
        if others.saturating_add(pages) > capacity {
            return Err(Error::Backpressure(format!(
                "managed KV incremental admission is waiting: request {} needs {pages} pages, {others} are claimed by other requests, capacity is {capacity}",
                request.id
            )));
        }
        claims.push((
            group.arena,
            u32::try_from(pages)
                .map_err(|_| Error::Overloaded("incremental page demand exceeds u32".into()))?,
        ));
    }
    let newly_installed = !state.capacity_claims.contains_key(session);
    state.capacity_claims.insert(session.clone(), claims);
    Ok(newly_installed)
}

fn full_context_sequence_capacity(plan: &ResolvedKvPlan, reach: u64) -> u64 {
    if reach == 0 {
        return 0;
    }
    plan.groups
        .iter()
        .map(|group| {
            let pages = reach.div_ceil(u64::from(group.page_tokens));
            u64::from(group.capacity_pages) / pages.max(1)
        })
        .min()
        .unwrap_or(0)
}

/// DS1.5: sentinel reason prefix for a probed prefix cursor that could not be
/// honored at prepare time (pages or the tensor snapshot were evicted between
/// admission and execution). The engine matches this reason, resets the
/// request's cursor, and replans a conventional zero-start prefill instead of
/// retrying a stale attach.
pub(crate) const MANAGED_PREFIX_CURSOR_LOST: &str = "managed prefix cursor lost";

fn managed_prefix_namespace(
    request: Option<&EngineCoreRequest>,
    runtime: &ManagedKvModelRuntime,
    cache_salt: Option<[u8; 32]>,
) -> Result<Option<KvPrefixNamespace>> {
    let (Some(request), Some(cache_salt)) = (request, cache_salt) else {
        return Ok(None);
    };
    let Some(binding) = request.execution_adapter_binding() else {
        return Ok(None);
    };
    if request.task_type != crate::engine::TaskType::Chat
        || request.model_instance_id() != Some(runtime.plan.model_instance)
        || binding.model_instance_id != runtime.plan.model_instance
    {
        return Ok(None);
    }

    // The current managed producer is text-only Qwen3. Each digest is derived
    // from exact lifecycle/adapter facts already sealed onto this request. The
    // model-generation fence deliberately prevents reuse across reloads until
    // artifact revisions and tokenizer ABIs become first-class lifecycle data.
    let model_revision = digest_parts(
        b"izwi.kv.loaded-model-generation.v1\0",
        &[&runtime.plan.model_instance.get().to_le_bytes()],
    );
    let adapter_abi = digest_parts(
        b"izwi.kv.loaded-adapter-abi.v1\0",
        &[
            &binding.adapter_instance_id.get().to_le_bytes(),
            &binding.adapter_abi_revision.get().to_le_bytes(),
            binding.capability_id.as_bytes(),
        ],
    );
    let tokenizer_or_input_encoding = digest_parts(
        b"izwi.kv.loaded-input-encoding.v1\0",
        &[
            &runtime.plan.model_instance.get().to_le_bytes(),
            &binding.adapter_instance_id.get().to_le_bytes(),
            binding.capability_id.as_bytes(),
        ],
    );
    let position_semantics = digest_parts(
        b"izwi.kv.position-semantics.v1\0",
        &[&runtime.plan.contract_fingerprint],
    );
    // One process serves many tenants behind a gateway. Without the tenant in
    // the salt, a tenant could time another tenant's prompts through prefix
    // hits, so a trusted tenant key scopes every page to its own namespace.
    let cache_salt = match request.tenant_key {
        Some(tenant_key) => digest_parts(
            b"izwi.kv.tenant-prefix-salt.v1\0",
            &[&cache_salt, &tenant_key],
        ),
        None => cache_salt,
    };
    Ok(Some(KvPrefixNamespace {
        model_instance: runtime.plan.model_instance,
        model_revision,
        adapter_abi,
        tokenizer_or_input_encoding,
        position_semantics,
        plan: runtime.plan.fingerprint(),
        multimodal_artifact: None,
        cache_salt,
    }))
}

/// Evenly divide one model's host-offload budget between its arenas so the
/// pools together never exceed the single budget charged to the model lease.
fn split_host_offload_budget(budget_bytes: u64, arenas: usize) -> u64 {
    budget_bytes / u64::try_from(arenas.max(1)).unwrap_or(u64::MAX)
}

fn digest_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn prefix_enabled_for_domain(
    contract: &InferenceStateContract,
    domain_id: crate::kv::CacheDomainId,
) -> bool {
    // Only paged-attention domains route through the token-page index below.
    // For a hybrid contract the paged attach is sound only when every other
    // domain can fork its committed state at the same cursor, so the strict
    // sharing gate must accept the whole contract; only paged-only contracts
    // keep the plain paged behavior. A ring (LFM2 ShortConv), append or
    // static domain has no snapshot fork path, so attaching pages beside it
    // would continue from a reset ring at cursor N.
    let paged_shareable = contract.domains.iter().any(|domain| {
        if domain.id() != domain_id {
            return false;
        }
        matches!(domain, StateDomainSpec::PagedAttention(spec) if matches!(
            spec.header.prefix,
            PrefixPolicy::CommittedPages { .. }
        ))
    });
    if !paged_shareable {
        return false;
    }
    let stateful_domains = contract
        .domains
        .iter()
        .filter(|domain| !matches!(domain, StateDomainSpec::PagedAttention(_)))
        .count();
    stateful_domains == 0 || tensor_snapshot_prefix_policy(contract).is_some()
}

fn sliding_window_for_domain(
    contract: &InferenceStateContract,
    domain_id: crate::kv::CacheDomainId,
) -> Result<Option<u32>> {
    let Some(domain) = contract
        .domains
        .iter()
        .find(|domain| domain.id() == domain_id)
    else {
        return Err(Error::InferenceError(
            "managed KV plan references a missing semantic domain".into(),
        ));
    };
    let StateDomainSpec::PagedAttention(spec) = domain else {
        return Ok(None);
    };
    let first = spec.layers.first().ok_or_else(|| {
        Error::InvalidInput("managed paged-attention domain has no layers".into())
    })?;
    if spec
        .layers
        .iter()
        .any(|layer| layer.pattern != first.pattern)
    {
        // A mixed local/global decoder (for example Gemma 3) shares one
        // physical page table across layers. Keep the complete logical table;
        // local layers lower their own bounded view at attention dispatch.
        return Ok(None);
    }
    Ok(match first.pattern {
        AttentionPattern::Full => None,
        AttentionPattern::SlidingWindow { window_tokens } => Some(window_tokens),
    })
}

fn validate_sliding_contract(
    contract: &InferenceStateContract,
    backend: BackendKind,
) -> Result<()> {
    for domain in &contract.domains {
        if sliding_window_for_domain(contract, domain.id())?.is_some() {
            // Every managed paged-attention backend carries the authoritative
            // first-page offset in its decode and prefill metadata. Keep this
            // match exhaustive so a future backend must make that contract an
            // explicit admission decision.
            match backend {
                BackendKind::Cpu | BackendKind::Metal | BackendKind::Cuda => {}
            }
        }
    }
    Ok(())
}

/// DS4: the longest committed prefix for one arena, continuing past the
/// device index into the host chain when the arena carries a host tail. The
/// digest chain stays complete across the tier boundary, so snapshot
/// reconciliation can walk a match back through host-resident pages.
fn managed_prefix_lookup(
    prefix_index: &mut CoordinatedPrefixIndex,
    host_chain: Option<&mut offload::HostChainIndex>,
    page_tokens: u32,
    namespace: &KvPrefixNamespace,
    tokens: &[u32],
    host_page_ceiling: usize,
) -> Result<KvPrefixMatch> {
    let telemetry = Arc::clone(prefix_index.telemetry());
    let matched = match host_chain {
        Some(chain) => offload::lookup_longest_with_host(
            prefix_index,
            chain,
            namespace,
            tokens,
            page_tokens,
            host_page_ceiling,
        )
        .map_err(prefix_error)?,
        None => prefix_index
            .lookup_longest(namespace, tokens, page_tokens)
            .map_err(prefix_error)?,
    };
    // Counted here, at preparation, and not in the index: admission probes
    // look up the same prompt first and would count every request twice.
    telemetry.record_prefix_lookup(usize_to_u64(tokens.len()), u64::from(matched.reused_tokens));
    Ok(matched)
}

/// DS4.3: outcome of restoring a matched host tail during prepare. The
/// restored boundary may sit below the matched end when a page failed to
/// copy; `record` carries the successfully restored span to the transaction's
/// commit.
struct PromotedTail {
    restored_tokens: u32,
    record: Option<PendingPrefixPromotion>,
}

/// Run one core-side KV page operation (zero, copy, growth, offload capture
/// or restore) on its arena's device thread instead of the tokio worker that
/// prepares the step.
fn on_device_thread<T: Send>(
    arena: &dyn KvArena,
    operation: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    DeviceThread::for_location(arena.device_location())?.run_sync(operation)?
}

/// Restores a matched host tail into the fresh device pages the current
/// transaction reserved for its span. The provisional table supplies one
/// block per tail page; a shorter table truncates the promotion exactly like
/// a failed copy does.
fn promote_host_tail(
    arena: &dyn KvArena,
    offload: &mut offload::ArenaOffload,
    arena_id: KvArenaId,
    page_tokens: u32,
    table: Option<&super::coordinator::GroupBlockTable>,
    tail: KvHostTailMatch,
) -> PromotedTail {
    let page = usize::try_from(page_tokens).unwrap_or(usize::MAX).max(1);
    let first_page = usize::try_from(tail.device_end_tokens).unwrap_or(usize::MAX) / page;
    let mut blocks = Vec::with_capacity(tail.digests.len());
    for index in first_page..first_page + tail.digests.len() {
        match table.and_then(|table| table.blocks.get(index)).copied() {
            Some(block) => blocks.push(block),
            None => break,
        }
    }
    // A failure to reach the device thread restores nothing: the match
    // falls back to its device-resident head.
    let copy = on_device_thread(arena, || {
        Ok(offload::promote_tail(
            arena,
            offload,
            &blocks,
            &tail.digests[..blocks.len()],
            &tail.slots[..blocks.len()],
        ))
    })
    .unwrap_or(offload::PromotionCopy {
        restored_pages: 0,
        latency_ns: 0,
    });
    let restored_tokens = tail.device_end_tokens.saturating_add(
        u32::try_from(copy.restored_pages.saturating_mul(page)).unwrap_or(u32::MAX),
    );
    let record = (copy.restored_pages > 0).then(|| PendingPrefixPromotion {
        arena: arena_id,
        page_tokens,
        device_end_tokens: tail.device_end_tokens,
        blocks: blocks[..copy.restored_pages].to_vec(),
        digests: tail.digests[..copy.restored_pages].to_vec(),
        latency_ns: copy.latency_ns,
    });
    PromotedTail {
        restored_tokens,
        record,
    }
}

fn prefix_publications(
    namespace: &KvPrefixNamespace,
    tokens: &[u32],
    page_tokens: u32,
    execution_start_tokens: u32,
    target_tokens: u32,
    reservation: &ManagedCacheDomainReservation,
    group: KvGroupId,
) -> Result<Vec<KvPrefixPublication>> {
    let page_tokens_usize = usize::try_from(page_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix page size exceeds usize".into()))?;
    let target = usize::try_from(target_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix target exceeds usize".into()))?;
    if target > tokens.len() || page_tokens_usize == 0 {
        return Ok(Vec::new());
    }
    let table = reservation
        .provisional_groups
        .iter()
        .find(|table| table.group == group)
        .ok_or_else(|| Error::InferenceError("managed prefix group table is missing".into()))?;
    let first_new_page = usize::try_from(execution_start_tokens / page_tokens)
        .map_err(|_| Error::InvalidInput("managed prefix start exceeds usize".into()))?;
    let complete_pages = target / page_tokens_usize;
    let mut previous = None;
    let mut publications = Vec::new();
    for page_index in 0..complete_pages {
        let start = page_index * page_tokens_usize;
        let key = KvPrefixPageKey::new(
            namespace,
            previous,
            start as u64,
            tokens[start..start + page_tokens_usize].to_vec(),
        )
        .map_err(prefix_error)?;
        previous = Some(key.digest());
        if page_index < first_new_page {
            continue;
        }
        let block = table.blocks.get(page_index).copied().ok_or_else(|| {
            Error::InferenceError("managed prefix publication exceeds its block table".into())
        })?;
        publications.push(KvPrefixPublication { key, block });
    }
    Ok(publications)
}

fn reservation_for_group(
    group: KvGroupId,
    page_tokens: u32,
    snapshot: &KvSnapshot,
    target_tokens: u32,
    shared_prefix: &[crate::kv::CacheBlockRef],
) -> Result<KvGroupReservation> {
    let required_pages = target_tokens
        .checked_add(page_tokens - 1)
        .ok_or_else(|| Error::Overloaded("managed KV page count overflow".into()))?
        / page_tokens;
    let required_pages = usize::try_from(required_pages)
        .map_err(|_| Error::Overloaded("managed KV page count exceeds usize".into()))?;
    let existing = snapshot
        .groups
        .iter()
        .find(|table| table.group == group)
        .map(|table| table.blocks.as_slice())
        .unwrap_or_default();
    let mut blocks = Vec::with_capacity(required_pages);
    if snapshot.committed_tokens == 0 {
        blocks.extend(shared_prefix.iter().copied().map(KvBlockIntent::Shared));
    } else if !shared_prefix.is_empty() {
        return Err(Error::InferenceError(
            "managed KV prefix pages cannot replace a committed request table".into(),
        ));
    }
    for (index, block) in existing.iter().take(required_pages).copied().enumerate() {
        let is_partial_tail = !snapshot.committed_tokens.is_multiple_of(page_tokens)
            && index + 1 == existing.len()
            && target_tokens > snapshot.committed_tokens;
        blocks.push(if is_partial_tail {
            KvBlockIntent::Writable(block)
        } else {
            KvBlockIntent::Existing(block)
        });
    }
    blocks.extend(std::iter::repeat_n(
        KvBlockIntent::Fresh,
        required_pages.saturating_sub(blocks.len()),
    ));
    Ok(KvGroupReservation { group, blocks })
}

/// DS1.2b: reconcile a paged prefix match with the committed tensor snapshot
/// index. The fork cursor must equal the paged attach cursor, so the matched
/// chain is walked back to the nearest boundary holding a committed snapshot;
/// a chain with no snapshot anywhere falls back to zero reuse rather than
/// attaching attention pages the tensor domains cannot follow.
#[allow(clippy::type_complexity)]
fn reconcile_tensor_snapshot_attach(
    sharing: &mut TensorSnapshotSharing,
    page_tokens: u32,
    prefix_match: &mut KvPrefixMatch,
    can_fork: bool,
    telemetry: &ManagedKvTelemetry,
) -> Result<Option<(u64, Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>)>> {
    if prefix_match.reused_tokens == 0 {
        return Ok(None);
    }
    if !can_fork {
        // The session already owns a live tensor sequence, so a fork would
        // cross a non-zero cursor. Only zero reuse is sound.
        telemetry.record_tensor_snapshot_truncation();
        prefix_match.blocks.clear();
        prefix_match.page_digests.clear();
        prefix_match.host_tail = None;
        prefix_match.reused_tokens = 0;
        return Ok(None);
    }
    while !prefix_match.page_digests.is_empty() {
        let boundary = u64::from(prefix_match.reused_tokens);
        let digest = prefix_match
            .page_digests
            .last()
            .copied()
            .expect("non-empty page digest chain");
        if let Some(domains) = sharing.index.lookup(digest, boundary)? {
            return Ok(Some((boundary, domains)));
        }
        prefix_match.pop_last_page(page_tokens);
        telemetry.record_tensor_snapshot_truncation();
    }
    Ok(None)
}

/// DS1.2b: build and index the committed snapshot for one aligned legacy
/// commit. `Ok(false)` means nothing new is held: the boundary was already
/// published identically, or the snapshot budget cannot hold it.
///
/// Every indexed snapshot is pinned in the tensor arena's committed backing,
/// so its bytes stay recorded as materialized on the model lease after the
/// publisher's sequence replaces or releases that state. Evicted snapshots
/// are unpinned and the lease reconciled (restoring their pending claim)
/// before their tensors drop.
fn publish_tensor_snapshot(
    state: &mut ManagedKvModelState,
    reservation: &ManagedCacheReservation,
    digest: [u8; 32],
    boundary: u64,
) -> Result<bool> {
    let Some(sharing) = state.tensor_snapshots.as_mut() else {
        return Ok(false);
    };
    let interval = u64::from(sharing.interval_tokens);
    if boundary == 0 || !boundary.is_multiple_of(interval) {
        return Ok(false);
    }
    let Some(clocked) = reservation.clocked_state.as_ref() else {
        return Ok(false);
    };
    let sequence = PhysicalStateSequenceId::new(clocked.sequence())?;
    let arena = state.runtime.tensor_state().cloned().ok_or_else(|| {
        Error::InferenceError("tensor snapshot publication lost its physical arena".into())
    })?;
    let mut entry = Vec::with_capacity(sharing.policy.tensor_domains.len());
    for domain in sharing.policy.tensor_domains.iter() {
        let snapshot = arena.read(sequence, *domain)?.ok_or_else(|| {
            Error::InferenceError(
                "committed tensor snapshot publication found no domain state".into(),
            )
        })?;
        if snapshot.cursor != boundary {
            return Err(Error::InferenceError(
                "committed tensor snapshot cursor disagrees with its paged boundary".into(),
            ));
        }
        entry.push((*domain, snapshot.components));
    }
    let domains: Arc<[(StateDomainId, Arc<[StateComponentValue]>)]> = entry.into();
    let byte_size = tensor_snapshot_backing_bytes(&domains);
    let publication = sharing
        .index
        .publish(digest, boundary, domains.clone(), byte_size)?;
    if publication.inserted {
        for (_, components) in domains.iter() {
            arena.pin_snapshot(components)?;
        }
    }
    for evicted in &publication.evicted {
        for (_, components) in evicted.domains.iter() {
            arena.unpin_snapshot(components)?;
        }
    }
    if !publication.evicted.is_empty() {
        reconcile_tensor_materialization(state);
    }
    drop(publication.evicted);
    Ok(publication.inserted)
}

/// Drop every indexed snapshot (model unload): unpin them, reconcile the
/// lease, then free them.
fn clear_tensor_snapshots(state: &mut ManagedKvModelState) -> Result<()> {
    let (Some(sharing), Some(arena)) = (
        state.tensor_snapshots.as_mut(),
        state.runtime.tensor_state().cloned(),
    ) else {
        return Ok(());
    };
    let cleared = sharing.index.clear();
    for entry in &cleared {
        for (_, components) in entry.domains.iter() {
            arena.unpin_snapshot(components)?;
        }
    }
    reconcile_tensor_materialization(state);
    drop(cleared);
    Ok(())
}

fn abort_domains(
    state: &mut ManagedKvModelState,
    txn_id: PlanId,
    domains: &[ManagedCacheDomainReservation],
) {
    for domain in domains {
        let _ = state
            .coordinators
            .get_mut(&domain.arena)
            .expect("reservation arena has a coordinator")
            .abort(txn_id);
    }
}

fn abort_reservation(state: &mut ManagedKvModelState, reservation: &ManagedCacheReservation) {
    state.pending_prefixes.remove(&reservation.txn_id);
    // DS4.3: an aborted transaction never publishes its restored pages, so
    // the host entries stay authoritative and keep their slots.
    state.pending_promotions.remove(&reservation.txn_id);
    state.exact_target_transactions.remove(&reservation.txn_id);
    abort_domains(state, reservation.txn_id, &reservation.domains);
    if reservation.clocked_state.is_some() {
        if let (Some(arena), Ok(transaction)) = (
            state.runtime.tensor_state(),
            PhysicalStateTransactionId::new(reservation.txn_id),
        ) {
            let _ = arena.abort(transaction);
        }
    }
}

fn arena_config(
    contract: &InferenceStateContract,
    group: &crate::kv::ResolvedKvGroup,
) -> Result<KvArenaConfig> {
    let domain = contract
        .domains
        .iter()
        .find(|domain| domain.id() == group.domain)
        .ok_or_else(|| {
            Error::InferenceError("resolved KV group lost its semantic domain".into())
        })?;
    let StateDomainSpec::PagedAttention(spec) = domain else {
        return Err(Error::InvalidInput(
            "dense paged KV arena requires a paged-attention domain".to_string(),
        ));
    };
    let mut layers = Vec::with_capacity(group.layers.len());
    for binding in &group.layers {
        let layer = spec
            .layers
            .iter()
            .find(|layer| layer.model_layer == binding.model_layer)
            .ok_or_else(|| Error::InferenceError("resolved KV layer binding is stale".into()))?;
        layers.push(KvLayerConfig {
            binding: *binding,
            num_kv_heads: layer.kv_heads,
            key_head_dim: layer.key_head_dim,
            value_head_dim: layer.value_head_dim,
        });
    }
    Ok(KvArenaConfig {
        id: group.arena,
        group: group.id,
        page_tokens: group.page_tokens,
        capacity_pages: group.capacity_pages,
        growth: match group.capacity_strategy {
            CapacityStrategy::Fixed { .. } => None,
            CapacityStrategy::AdmissionGrowable {
                initial_blocks,
                growth_quantum,
                ..
            } => Some(KvArenaGrowthConfig {
                initial_pages: initial_blocks,
                growth_quantum_pages: growth_quantum,
            }),
            CapacityStrategy::BoundedLazy { .. } | CapacityStrategy::Reserved { .. } => {
                return Err(Error::InferenceError(
                    "resolved paged KV plan contains a non-arena capacity strategy".into(),
                ));
            }
        },
        dtype: candle_dtype(group.storage.dtype())?,
        layers,
    })
}

fn candle_dtype(dtype: KvStorageDType) -> Result<DType> {
    match dtype {
        KvStorageDType::F32 => Ok(DType::F32),
        KvStorageDType::F16 => Ok(DType::F16),
        KvStorageDType::Bf16 => Ok(DType::BF16),
        KvStorageDType::I64 | KvStorageDType::I8 | KvStorageDType::Q4 => Err(Error::InvalidInput(
            "dense KV arena cannot allocate quantized storage".to_string(),
        )),
    }
}

pub(crate) fn coordinator_error(error: impl fmt::Display) -> Error {
    Error::InferenceError(format!(
        "managed KV coordinator rejected transaction: {error}"
    ))
}

fn prefix_error(error: impl fmt::Display) -> Error {
    Error::InferenceError(format!(
        "managed KV prefix index rejected operation: {error}"
    ))
}

#[cfg(test)]
#[path = "managed_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "managed_stress.rs"]
mod stress_tests;

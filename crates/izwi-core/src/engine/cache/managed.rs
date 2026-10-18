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

mod binding;
mod capacity;
mod finalize;
mod prefix;
mod prepare;
mod snapshots;
#[cfg(test)]
use binding::reserve_managed_arena;
use binding::{
    ensure_capacity_claim, ensure_chunked_capacity_claim, ensure_incremental_capacity_claim,
    ensure_session_tables, full_context_sequence_capacity, reconcile_tensor_materialization,
    release_tensor_sequence, validate_reservation_session_generation,
};
#[cfg(test)]
use capacity::{
    cuda_contiguous_replacement_peak_pages, cuda_contiguous_replacement_required_bytes,
    cuda_largest_contiguous_replacement_extra, cuda_resident_required_bytes,
    fit_cuda_contiguous_token_reach, fit_cuda_resident_token_reach, fit_resident_rows_and_reach,
    managed_paged_capacity_strategy, CudaContiguousPagedGeometry,
};
use capacity::{
    cuda_managed_state_peak_authorization, managed_state_resources,
    plan_managed_state_capacity_with_policy, prefix_index_pages, tensor_state_resources,
};
use prefix::{
    clear_tensor_snapshots, managed_prefix_lookup, managed_prefix_namespace, on_device_thread,
    prefix_enabled_for_domain, prefix_error, prefix_publications, promote_host_tail,
    publish_tensor_snapshot, reconcile_tensor_snapshot_attach, sliding_window_for_domain,
    split_host_offload_budget, validate_sliding_contract, PromotedTail,
};
use prepare::abort_reservation;
#[cfg(test)]
use prepare::reservation_for_group;
use snapshots::usize_to_u64;

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

    pub(crate) fn worker_backend(&self) -> BackendKind {
        self.worker_backend
    }
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

/// DS1.5: sentinel reason prefix for a probed prefix cursor that could not be
/// honored at prepare time (pages or the tensor snapshot were evicted between
/// admission and execution). The engine matches this reason, resets the
/// request's cursor, and replans a conventional zero-start prefill instead of
/// retrying a stale attach.
pub(crate) const MANAGED_PREFIX_CURSOR_LOST: &str = "managed prefix cursor lost";

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

#[cfg(test)]
#[path = "managed_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "managed_stress.rs"]
mod stress_tests;

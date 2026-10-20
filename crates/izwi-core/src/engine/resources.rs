//! Backend-neutral resource estimates, reservations, and reconciliation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backends::BackendKind;
use crate::error::{Error, Result};

/// A resource quantity whose capacity may be unavailable from the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResourceAmount {
    #[default]
    Unknown,
    Known(u64),
}

impl ResourceAmount {
    pub const fn known(value: u64) -> Self {
        Self::Known(value)
    }

    fn checked_add(self, other: Self) -> Result<Self> {
        match (self, other) {
            (Self::Known(left), Self::Known(right)) => left
                .checked_add(right)
                .map(Self::Known)
                .ok_or_else(|| Error::Overloaded("resource accounting overflow".to_string())),
            _ => Ok(Self::Unknown),
        }
    }

    fn checked_sub(self, other: Self) -> Result<Self> {
        match (self, other) {
            (Self::Known(left), Self::Known(right)) => left
                .checked_sub(right)
                .map(Self::Known)
                .ok_or_else(|| Error::InferenceError("resource ledger underflow".to_string())),
            _ => Ok(Self::Unknown),
        }
    }

    fn fits(self, capacity: Self) -> bool {
        match (self, capacity) {
            (Self::Known(requested), Self::Known(capacity)) => requested <= capacity,
            // Unknown capacity is never treated as infinite: callers must configure
            // a concrete cap before reserving a known quantity in that domain.
            (Self::Known(0), Self::Unknown) | (Self::Unknown, _) => true,
            (Self::Known(_), Self::Unknown) => false,
        }
    }

    fn positive_growth_over(self, current: Self) -> Result<Self> {
        match (self, current) {
            (Self::Known(next), Self::Known(current)) => {
                Ok(Self::Known(next.saturating_sub(current)))
            }
            _ => Err(Error::InvalidInput(
                "resource resize contains an unresolved quantity".to_string(),
            )),
        }
    }
}

/// Resource vector used for estimates, capacity, and observed usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceVector {
    pub host_bytes: ResourceAmount,
    pub device_bytes: ResourceAmount,
    pub unified_bytes: ResourceAmount,
    pub kv_bytes: ResourceAmount,
    pub temporary_bytes: ResourceAmount,
    pub compute_slots: ResourceAmount,
}

impl ResourceVector {
    pub const fn zero() -> Self {
        Self {
            host_bytes: ResourceAmount::Known(0),
            device_bytes: ResourceAmount::Known(0),
            unified_bytes: ResourceAmount::Known(0),
            kv_bytes: ResourceAmount::Known(0),
            temporary_bytes: ResourceAmount::Known(0),
            compute_slots: ResourceAmount::Known(0),
        }
    }

    /// Backend-neutral transient memory. Physical planning resolves this into
    /// host, unified, or device memory before reservation.
    pub const fn temporary_workspace(bytes: u64) -> Self {
        Self {
            temporary_bytes: ResourceAmount::Known(bytes),
            ..Self::zero()
        }
    }

    pub fn checked_add(self, other: Self) -> Result<Self> {
        Ok(Self {
            host_bytes: self.host_bytes.checked_add(other.host_bytes)?,
            device_bytes: self.device_bytes.checked_add(other.device_bytes)?,
            unified_bytes: self.unified_bytes.checked_add(other.unified_bytes)?,
            kv_bytes: self.kv_bytes.checked_add(other.kv_bytes)?,
            temporary_bytes: self.temporary_bytes.checked_add(other.temporary_bytes)?,
            compute_slots: self.compute_slots.checked_add(other.compute_slots)?,
        })
    }

    pub fn checked_sub(self, other: Self) -> Result<Self> {
        Ok(Self {
            host_bytes: self.host_bytes.checked_sub(other.host_bytes)?,
            device_bytes: self.device_bytes.checked_sub(other.device_bytes)?,
            unified_bytes: self.unified_bytes.checked_sub(other.unified_bytes)?,
            kv_bytes: self.kv_bytes.checked_sub(other.kv_bytes)?,
            temporary_bytes: self.temporary_bytes.checked_sub(other.temporary_bytes)?,
            compute_slots: self.compute_slots.checked_sub(other.compute_slots)?,
        })
    }

    pub fn fits_within(self, capacity: Self) -> bool {
        self.host_bytes.fits(capacity.host_bytes)
            && self.device_bytes.fits(capacity.device_bytes)
            && self.unified_bytes.fits(capacity.unified_bytes)
            && self.kv_bytes.fits(capacity.kv_bytes)
            && self.temporary_bytes.fits(capacity.temporary_bytes)
            && self.compute_slots.fits(capacity.compute_slots)
    }

    pub fn is_fully_known(self) -> bool {
        [
            self.host_bytes,
            self.device_bytes,
            self.unified_bytes,
            self.kv_bytes,
            self.temporary_bytes,
            self.compute_slots,
        ]
        .into_iter()
        .all(|amount| matches!(amount, ResourceAmount::Known(_)))
    }

    /// Return the total transient workspace represented across memory domains.
    /// Persistent KV and compute-slot quantities are not valid batch scratch.
    pub fn workspace_bytes(self) -> Result<u64> {
        if self.kv_bytes != ResourceAmount::Known(0)
            || self.compute_slots != ResourceAmount::Known(0)
        {
            return Err(Error::InvalidInput(
                "batch workspace cannot contain persistent KV or compute-slot resources"
                    .to_string(),
            ));
        }
        [
            self.host_bytes,
            self.device_bytes,
            self.unified_bytes,
            self.temporary_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, amount| match amount {
            ResourceAmount::Known(value) => total
                .checked_add(value)
                .ok_or_else(|| Error::Overloaded("batch workspace total overflow".to_string())),
            ResourceAmount::Unknown => Err(Error::InvalidInput(
                "batch workspace contains an unresolved quantity".to_string(),
            )),
        })
    }

    /// Per domain, how far `self` exceeds `current` (zero where it does not).
    pub(crate) fn positive_growth_over(self, current: Self) -> Result<Self> {
        Ok(Self {
            host_bytes: self.host_bytes.positive_growth_over(current.host_bytes)?,
            device_bytes: self
                .device_bytes
                .positive_growth_over(current.device_bytes)?,
            unified_bytes: self
                .unified_bytes
                .positive_growth_over(current.unified_bytes)?,
            kv_bytes: self.kv_bytes.positive_growth_over(current.kv_bytes)?,
            temporary_bytes: self
                .temporary_bytes
                .positive_growth_over(current.temporary_bytes)?,
            compute_slots: self
                .compute_slots
                .positive_growth_over(current.compute_slots)?,
        })
    }
}

pub type ResourceEstimate = ResourceVector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReservationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceReservation {
    pub id: ReservationId,
    pub resources: ResourceVector,
}

/// Transactional resource ledger. Failed reservations never mutate usage.
#[derive(Debug)]
pub struct ResourceLedger {
    capacity: ResourceVector,
    used: ResourceVector,
    reservations: HashMap<ReservationId, ResourceVector>,
    next_id: u64,
}

impl ResourceLedger {
    pub fn new(capacity: ResourceVector) -> Self {
        Self {
            capacity,
            used: ResourceVector::zero(),
            reservations: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn capacity(&self) -> ResourceVector {
        self.capacity
    }

    pub fn used(&self) -> ResourceVector {
        self.used
    }

    fn update_capacity(&mut self, capacity: ResourceVector) {
        // Existing reservations remain owned even if a live provider lowers
        // its advertised ceiling. Future guarded admission must observe the
        // latest ceiling and fail until usage returns beneath it.
        self.capacity = capacity;
    }

    pub fn reserve(&mut self, resources: ResourceVector) -> Result<ResourceReservation> {
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "resource reservation contains an unresolved quantity".to_string(),
            ));
        }
        let used = self.used.checked_add(resources)?;
        if !used.fits_within(self.capacity) {
            return Err(Error::Overloaded(
                "requested resources exceed available capacity".to_string(),
            ));
        }
        let id = ReservationId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.reservations.insert(id, resources);
        self.used = used;
        Ok(ResourceReservation { id, resources })
    }

    pub fn release(&mut self, id: ReservationId) -> Result<bool> {
        let Some(resources) = self.reservations.remove(&id) else {
            return Ok(false);
        };
        self.used = self.used.checked_sub(resources)?;
        Ok(true)
    }

    pub fn resize(&mut self, id: ReservationId, resources: ResourceVector) -> Result<bool> {
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "resource reservation contains an unresolved quantity".to_string(),
            ));
        }
        let Some(current) = self.reservations.get(&id).copied() else {
            return Ok(false);
        };
        let used = self.used.checked_sub(current)?.checked_add(resources)?;
        if !used.fits_within(self.capacity) {
            return Err(Error::Overloaded(
                "resized resources exceed available capacity".to_string(),
            ));
        }
        self.reservations.insert(id, resources);
        self.used = used;
        Ok(true)
    }

    /// Replace a reservation with one no larger in any domain. Shrinking only
    /// lowers `used`, so it holds whatever the current capacity is.
    fn shrink(&mut self, id: ReservationId, resources: ResourceVector) -> Result<bool> {
        let Some(current) = self.reservations.get(&id).copied() else {
            return Ok(false);
        };
        if !resources.is_fully_known() || !resources.fits_within(current) {
            return Err(Error::InvalidInput(
                "a reservation shrink must be known and no larger than the reservation".to_string(),
            ));
        }
        self.used = self.used.checked_sub(current)?.checked_add(resources)?;
        self.reservations.insert(id, resources);
        Ok(true)
    }

    fn reservation(&self, id: ReservationId) -> Option<ResourceVector> {
        self.reservations.get(&id).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationClass {
    Model,
    Request,
    Cache,
    Pipeline,
    BatchWorkspace,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReservationOwner {
    pub class: ReservationClass,
    pub key: String,
}

impl ReservationOwner {
    pub fn new(class: ReservationClass, key: impl Into<String>) -> Self {
        Self {
            class,
            key: key.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacitySource {
    OperatingSystem,
    MetalWorkingSet,
    CudaDriver,
    Test,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalCapacitySnapshot {
    /// Total resource budget controlled by the authority. This is the ceiling
    /// for the complete reservation ledger, independent of whether a lease has
    /// materialized into a physical allocation yet.
    pub capacity: ResourceVector,
    /// Live physical headroom available to a *new* allocation at the instant
    /// the snapshot is taken. Providers backed by an OS or device driver must
    /// report actual free/reclaimable capacity here; allocations belonging to
    /// existing leases may therefore already be subtracted from this value.
    ///
    /// `ResourceAuthority` compares the new reservation plus every
    /// unmaterialized claim against this vector. It separately compares the
    /// complete reservation ledger against `capacity`, avoiding
    /// double-counting materialized leases.
    pub available: ResourceVector,
    pub source: CapacitySource,
}

pub trait PhysicalCapacityProvider: std::fmt::Debug + Send + Sync {
    fn snapshot(&self) -> PhysicalCapacitySnapshot;

    /// Optional operator-configured ceiling, distinct from volatile physical
    /// availability. Advisory shared-memory authorities still enforce this
    /// limit against their complete reservation ledger.
    fn configured_budget(&self) -> Option<ResourceVector> {
        None
    }

    /// Refresh physical headroom after an owner has dropped device-backed
    /// allocations. Cached providers override this so the next admission does
    /// not inherit a pre-release sample.
    fn refresh_after_release(&self) -> PhysicalCapacitySnapshot {
        self.snapshot()
    }

    /// Bytes this process has allocated on the inference device right now,
    /// sampled fresh: Metal's `currentAllocatedSize`, or CUDA's used device
    /// memory (total less free, read from the driver without touching
    /// cudarc's shared error state). `None` when the backend has no device
    /// observer (CPU), which skips observed reconciliation.
    fn observed_device_usage_bytes(&self) -> Option<u64> {
        None
    }

    /// How host memory gates Metal work, for a Metal provider; `None` for
    /// every other provider.
    fn metal_load_admission(&self) -> Option<MetalLoadAdmission> {
        None
    }

    /// The host memory the operating system reports available right now,
    /// sampled fresh, when this provider reads a host figure that does not
    /// already bound its `available` vector (a Metal provider whose host
    /// sample is advisory). `None` otherwise.
    fn advisory_host_available_bytes(&self) -> Option<u64> {
        None
    }
}

/// How the host's available memory gates Metal loads and requests
/// (`IZWI_METAL_LOAD_ADMISSION`). The GPU working set
/// (`recommendedMaxWorkingSetSize` less this process's allocations) always
/// gates: past it a command buffer fails with an out-of-memory error. Host
/// memory is a softer limit on Apple silicon: macOS compresses and swaps
/// other processes to make room, which is what llama.cpp and Ollama rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalLoadAdmission {
    /// Loads and requests also need the host's available memory.
    Strict,
    /// Only the GPU working set gates; a load larger than the host's
    /// available memory is logged and reported, not refused.
    #[default]
    Advisory,
    /// No live-memory check before a load; the ledger budget still applies,
    /// and requests still need the GPU working set.
    Off,
}

impl MetalLoadAdmission {
    pub const ENV: &'static str = "IZWI_METAL_LOAD_ADMISSION";

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "strict" => Some(Self::Strict),
            "advisory" => Some(Self::Advisory),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    /// The operator's setting; unset is [`Self::Advisory`], anything that
    /// does not parse is a startup error.
    pub fn from_env() -> Result<Self> {
        match std::env::var(Self::ENV) {
            Ok(raw) => Self::parse(&raw).ok_or_else(|| {
                Error::ConfigError(format!(
                    "{} must be strict, advisory or off, not {raw:?}",
                    Self::ENV
                ))
            }),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(err) => Err(Error::ConfigError(format!(
                "failed to read {}: {err}",
                Self::ENV
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Advisory => "advisory",
            Self::Off => "off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceAuthoritySnapshot {
    pub physical: PhysicalCapacitySnapshot,
    pub reserved: ResourceVector,
    /// The part of `reserved` its owners report as allocated, so already
    /// excluded from `physical.available`.
    pub materialized: ResourceVector,
    /// The part of `reserved` not yet allocated, which guarded admission
    /// charges against `physical.available`.
    pub pending: ResourceVector,
    pub reservations: usize,
}

#[derive(Debug)]
struct AuthorityState {
    ledger: ResourceLedger,
    owners: HashMap<ReservationId, ReservationOwner>,
    /// Portion of each reservation that is already visible in the physical
    /// provider's used-memory reading. The remainder is pending allocation and
    /// must be subtracted from live headroom before admitting more work.
    materialized: HashMap<ReservationId, ResourceVector>,
    poisoned: Option<String>,
}

/// A fixed-size diagnostic: reservation keys can contain caller-controlled
/// identifiers and must never be included in capacity rejection summaries.
/// The per-class `reserved`/`materialized` splits exist so a rejection names
/// which residency actually holds the device, not just unmaterialized work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingClassSummary {
    class: ReservationClass,
    reservations: usize,
    reserved: ResourceVector,
    materialized: ResourceVector,
    pending: ResourceVector,
}

impl AuthorityState {
    fn materialization_exceeded(
        &self,
        id: ReservationId,
        operation: &str,
        authorized: ResourceVector,
        observed: ResourceVector,
    ) -> Error {
        let exceeded_domains: Vec<_> = [
            ("host_bytes", observed.host_bytes, authorized.host_bytes),
            (
                "device_bytes",
                observed.device_bytes,
                authorized.device_bytes,
            ),
            (
                "unified_bytes",
                observed.unified_bytes,
                authorized.unified_bytes,
            ),
            ("kv_bytes", observed.kv_bytes, authorized.kv_bytes),
            (
                "temporary_bytes",
                observed.temporary_bytes,
                authorized.temporary_bytes,
            ),
            (
                "compute_slots",
                observed.compute_slots,
                authorized.compute_slots,
            ),
        ]
        .into_iter()
        .filter_map(|(domain, usage, limit)| (!usage.fits(limit)).then_some(domain))
        .collect();
        // Owner keys can contain arbitrary caller data. The authority's numeric
        // reservation ID and class identify the owner without copying its key.
        let owner_class = self.owners.get(&id).map(|owner| owner.class);
        Error::InferenceError(format!(
            "materialized resource usage exceeds its authorized reservation: reservation_id={}, owner_class={owner_class:?}, operation={operation}, authorized={authorized:?}, observed={observed:?}, exceeded_domains={exceeded_domains:?}",
            id.0,
        ))
    }

    fn pending_by_class(
        &self,
        excluded: Option<ReservationId>,
    ) -> Result<[PendingClassSummary; 5]> {
        let mut summaries = [
            ReservationClass::Model,
            ReservationClass::Request,
            ReservationClass::Cache,
            ReservationClass::Pipeline,
            ReservationClass::BatchWorkspace,
        ]
        .map(|class| PendingClassSummary {
            class,
            reservations: 0,
            reserved: ResourceVector::zero(),
            materialized: ResourceVector::zero(),
            pending: ResourceVector::zero(),
        });
        for (id, resources) in &self.ledger.reservations {
            if excluded == Some(*id) {
                continue;
            }
            let owner = self.owners.get(id).ok_or_else(|| {
                Error::InferenceError("resource reservation owner is missing".into())
            })?;
            let index = match owner.class {
                ReservationClass::Model => 0,
                ReservationClass::Request => 1,
                ReservationClass::Cache => 2,
                ReservationClass::Pipeline => 3,
                ReservationClass::BatchWorkspace => 4,
            };
            let materialized = self
                .materialized
                .get(id)
                .copied()
                .unwrap_or_else(ResourceVector::zero);
            summaries[index].reservations += 1;
            summaries[index].reserved = summaries[index].reserved.checked_add(*resources)?;
            summaries[index].materialized = summaries[index]
                .materialized
                .checked_add(materialized)?;
            summaries[index].pending = summaries[index]
                .pending
                .checked_add(resources.positive_growth_over(materialized)?)?;
        }
        Ok(summaries)
    }

    fn pending_resources(&self) -> Result<ResourceVector> {
        self.ledger.reservations.iter().try_fold(
            ResourceVector::zero(),
            |pending, (id, resources)| {
                let materialized = self
                    .materialized
                    .get(id)
                    .copied()
                    .unwrap_or_else(ResourceVector::zero);
                pending.checked_add(resources.positive_growth_over(materialized)?)
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapacityEnforcement {
    Guarded,
    Advisory,
}

/// One transactional authority for every physical-memory consumer on a backend.
#[derive(Debug)]
pub struct ResourceAuthority {
    provider: Arc<dyn PhysicalCapacityProvider>,
    domain_policy: ResourceDomainPolicy,
    capacity_enforcement: CapacityEnforcement,
    state: Mutex<AuthorityState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceDomainPolicy {
    Independent,
    /// CPU host allocations and Metal unified allocations consume the same
    /// physical memory on a unified-memory host. Canonicalize both into the
    /// unified ledger domain so independently constructed workers cannot spend
    /// the process memory budget twice.
    SharedHostUnified,
}

impl ResourceAuthority {
    pub fn new(provider: Arc<dyn PhysicalCapacityProvider>) -> Self {
        Self::with_policies(
            provider,
            ResourceDomainPolicy::Independent,
            CapacityEnforcement::Guarded,
        )
    }

    /// Keep complete reservation accounting without rejecting allocations from
    /// volatile physical-capacity samples. CPU and unified-memory Metal share
    /// this authority so actual host/backend allocation remains authoritative;
    /// discrete CUDA authorities continue to use [`Self::new`].
    pub(crate) fn new_advisory_shared_host_unified(
        provider: Arc<dyn PhysicalCapacityProvider>,
    ) -> Self {
        Self::with_policies(
            provider,
            ResourceDomainPolicy::SharedHostUnified,
            CapacityEnforcement::Advisory,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_advisory(provider: Arc<dyn PhysicalCapacityProvider>) -> Self {
        Self::with_policies(
            provider,
            ResourceDomainPolicy::Independent,
            CapacityEnforcement::Advisory,
        )
    }

    fn with_policies(
        provider: Arc<dyn PhysicalCapacityProvider>,
        domain_policy: ResourceDomainPolicy,
        capacity_enforcement: CapacityEnforcement,
    ) -> Self {
        let capacity = match capacity_enforcement {
            CapacityEnforcement::Guarded => {
                normalize_physical_resource_domains(provider.snapshot().capacity, domain_policy)
            }
            CapacityEnforcement::Advisory => {
                advisory_ledger_capacity(provider.configured_budget(), domain_policy)
            }
        };
        Self {
            provider,
            domain_policy,
            capacity_enforcement,
            state: Mutex::new(AuthorityState {
                ledger: ResourceLedger::new(capacity),
                owners: HashMap::new(),
                materialized: HashMap::new(),
                poisoned: None,
            }),
        }
    }

    pub fn snapshot(&self) -> ResourceAuthoritySnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let physical = self.normalized_physical_snapshot();
        // An accounting overflow reads as unknown rather than failing a probe.
        let materialized = state
            .materialized
            .values()
            .try_fold(ResourceVector::zero(), |total, usage| {
                total.checked_add(*usage)
            })
            .unwrap_or_default();
        ResourceAuthoritySnapshot {
            physical,
            reserved: state.ledger.used(),
            materialized,
            pending: state.pending_resources().unwrap_or_default(),
            reservations: state.owners.len(),
        }
    }

    /// Permanently fail new work after a backend-fatal asynchronous error.
    /// Metal command-buffer OOM can leave queued command/fence bookkeeping in
    /// an unusable state, so process/device recreation is required — until
    /// the backend fully drains, which lets [`Self::clear_poison`] restore
    /// fresh loads.
    pub(crate) fn poison(&self, reason: impl Into<String>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.poisoned.is_some() {
            return;
        }
        let reason = reason.into();
        state.poisoned = Some(reason.clone());
        drop(state);
        // One canonical event for operators and the desktop monitor; the
        // per-request errors wrap this reason afterwards.
        tracing::error!(
            reason = %reason,
            "backend resource authority poisoned; new physical reservations and model loads are rejected until the backend is recreated or fully drains"
        );
    }

    pub(crate) fn poison_reason(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .poisoned
            .clone()
    }

    /// Clear the poison marker once the backend has been fully drained. With
    /// no resident models and no active physical work, nothing queued before
    /// the failure can be misattributed to fresh loads, so the fail-closed
    /// guard is no longer needed — the in-process equivalent of llama.cpp's
    /// "recreate the backend to recover". Returns whether a poison cleared.
    pub(crate) fn clear_poison(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(reason) = state.poisoned.take() else {
            return false;
        };
        drop(state);
        tracing::info!(
            reason = %reason,
            "backend resource authority recovered after the backend fully drained"
        );
        true
    }

    /// Stable backend planning headroom for load-time sizing. Unlike guarded
    /// admission this deliberately ignores a volatile live-available sample:
    /// CPU compression/swap and Metal working-set accounting make that sample
    /// unsuitable as a context-size contract. Explicit operator budgets remain
    /// hard and are preferred over the physical total.
    pub(crate) fn planning_headroom(&self) -> Result<ResourceVector> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(reason) = state.poisoned.as_ref() {
            return Err(Error::InferenceError(format!(
                "backend resource authority is poisoned and must be recreated: {reason}"
            )));
        }
        let ceiling = match self.provider.configured_budget() {
            Some(budget) => normalize_resource_domains(budget, self.domain_policy)?,
            None => self.normalized_physical_snapshot().capacity,
        };
        ceiling.positive_growth_over(state.ledger.used())
    }

    /// Return planning headroom in the backend's canonical memory domain.
    /// CPU and Metal share one unified ledger in production, so CPU must not
    /// read the now-zero host alias from that normalized vector. Guarded CUDA
    /// must additionally respect live free device memory: checkpoint residency
    /// can exceed its estimated ledger claim, and deferred state growth is
    /// admitted against the same physical observation later.
    pub(crate) fn planning_headroom_bytes(&self, backend: BackendKind) -> Result<ResourceAmount> {
        if self.capacity_enforcement == CapacityEnforcement::Guarded && backend == BackendKind::Cuda
        {
            let state = self.state.lock().map_err(|_| {
                Error::InferenceError("resource authority mutex poisoned".to_string())
            })?;
            if let Some(reason) = state.poisoned.as_ref() {
                return Err(Error::InferenceError(format!(
                    "backend resource authority is poisoned and must be recreated: {reason}"
                )));
            }
            let physical = self.normalized_physical_snapshot();
            let ceiling = match self.provider.configured_budget() {
                Some(budget) => normalize_resource_domains(budget, self.domain_policy)?,
                None => physical.capacity,
            };
            let ledger_headroom = ceiling.positive_growth_over(state.ledger.used())?;
            let live_headroom = physical
                .available
                .positive_growth_over(state.pending_resources()?)?;
            return Ok(
                match (ledger_headroom.device_bytes, live_headroom.device_bytes) {
                    (ResourceAmount::Known(ledger), ResourceAmount::Known(live)) => {
                        ResourceAmount::Known(ledger.min(live))
                    }
                    _ => ResourceAmount::Unknown,
                },
            );
        }

        let headroom = self.planning_headroom()?;
        Ok(match (self.domain_policy, backend) {
            (ResourceDomainPolicy::SharedHostUnified, BackendKind::Cpu | BackendKind::Metal) => {
                headroom.unified_bytes
            }
            (_, BackendKind::Cpu) => headroom.host_bytes,
            (_, BackendKind::Metal) => headroom.unified_bytes,
            (_, BackendKind::Cuda) => headroom.device_bytes,
        })
    }

    /// Headroom a load-time ASR state fit may spend, in the backend's domain:
    /// planning headroom, clamped on CPU and Metal to the live sample (less
    /// pending claims) when the provider has one (ADM-7). Planning headroom
    /// alone is a ledger figure: on a machine whose memory is already in use
    /// it lets a fit allocate state that the next request then cannot run
    /// beside. Everything reserved before the fit, such as the model's own
    /// lease and a deferred request envelope, is charged to both figures, so
    /// a fit can never spend it. CUDA's planning headroom already takes the
    /// live minimum.
    pub(crate) fn state_fit_headroom_bytes(&self, backend: BackendKind) -> Result<ResourceAmount> {
        let planning = self.planning_headroom_bytes(backend)?;
        if backend == BackendKind::Cuda {
            return Ok(planning);
        }
        Ok(match (planning, self.live_headroom_in_domain(backend)?) {
            (ResourceAmount::Known(planning), Some(live)) => {
                ResourceAmount::Known(planning.min(live))
            }
            (planning, _) => planning,
        })
    }

    /// Fresh device usage from the provider's observer; see
    /// [`PhysicalCapacityProvider::observed_device_usage_bytes`].
    pub(crate) fn observed_device_usage_bytes(&self) -> Option<u64> {
        self.provider.observed_device_usage_bytes()
    }

    /// Reserved bytes in `backend`'s canonical domain (every lease).
    pub(crate) fn reserved_bytes_in_domain(&self, backend: BackendKind) -> u64 {
        let reserved = self.snapshot().reserved;
        known_bytes(self.domain_amount(reserved, backend))
    }

    /// Bytes the reservations of `classes` hold, in `backend`'s domain.
    pub(crate) fn reserved_bytes_for_classes(
        &self,
        backend: BackendKind,
        classes: &[ReservationClass],
    ) -> u64 {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state
            .ledger
            .reservations
            .iter()
            .filter(|(id, _)| {
                state
                    .owners
                    .get(id)
                    .is_some_and(|owner| classes.contains(&owner.class))
            })
            .map(|(_, resources)| known_bytes(self.domain_amount(*resources, backend)))
            .fold(0, u64::saturating_add)
    }

    /// The ceiling planning headroom is measured against, in `backend`'s
    /// domain: the operator's or derived budget, else physical capacity.
    pub(crate) fn planning_ceiling_bytes(&self, backend: BackendKind) -> Option<u64> {
        let ceiling = match self.provider.configured_budget() {
            Some(budget) => normalize_resource_domains(budget, self.domain_policy).ok()?,
            None => self.normalized_physical_snapshot().capacity,
        };
        match self.domain_amount(ceiling, backend) {
            ResourceAmount::Known(bytes) => Some(bytes),
            ResourceAmount::Unknown => None,
        }
    }

    /// Device memory the process holds outside every lease's materialized
    /// usage, in `backend`'s domain: observed device usage less what the
    /// ledger records as allocated (Candle pool slack, power-of-two rounding
    /// of activations, command buffers, lazily grown backend contexts).
    /// `None` without a device observer.
    pub(crate) fn untracked_device_bytes(&self, backend: BackendKind) -> Option<u64> {
        let observed = self.observed_device_usage_bytes()?;
        let materialized = self.snapshot().materialized;
        Some(observed.saturating_sub(known_bytes(self.domain_amount(materialized, backend))))
    }

    /// The one safety reserve ASR state fits leave free: the operator's
    /// configured floor, or the measured untracked device memory when that is
    /// larger. Untracked memory is what this process allocated outside any
    /// lease so far; leaving at least as much beside the fitted state lets the
    /// same growth recur without crowding the requests the state serves.
    pub(crate) fn state_safety_reserve_bytes(
        &self,
        backend: BackendKind,
        configured_floor_bytes: u64,
    ) -> u64 {
        self.untracked_device_bytes(backend)
            .map_or(configured_floor_bytes, |untracked| {
                untracked.max(configured_floor_bytes)
            })
    }

    fn domain_amount(&self, resources: ResourceVector, backend: BackendKind) -> ResourceAmount {
        match (self.domain_policy, backend) {
            (ResourceDomainPolicy::SharedHostUnified, BackendKind::Cpu | BackendKind::Metal) => {
                resources.unified_bytes
            }
            (_, BackendKind::Cpu) => resources.host_bytes,
            (_, BackendKind::Metal) => resources.unified_bytes,
            (_, BackendKind::Cuda) => resources.device_bytes,
        }
    }

    /// The live sample less pending claims, in `backend`'s canonical domain,
    /// or `None` when the provider has no trustworthy sample.
    fn live_headroom_in_domain(&self, backend: BackendKind) -> Result<Option<u64>> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(reason) = state.poisoned.as_ref() {
            return Err(Error::InferenceError(format!(
                "backend resource authority is poisoned and must be recreated: {reason}"
            )));
        }
        let physical = self.normalized_physical_snapshot();
        if physical.source == CapacitySource::Unavailable {
            return Ok(None);
        }
        let live = physical
            .available
            .positive_growth_over(state.pending_resources()?)?;
        let amount = match (self.domain_policy, backend) {
            (ResourceDomainPolicy::SharedHostUnified, BackendKind::Cpu | BackendKind::Metal) => {
                live.unified_bytes
            }
            (_, BackendKind::Cpu) => live.host_bytes,
            (_, BackendKind::Metal) => live.unified_bytes,
            (_, BackendKind::Cuda) => live.device_bytes,
        };
        Ok(match amount {
            ResourceAmount::Known(bytes) => Some(bytes),
            ResourceAmount::Unknown => None,
        })
    }

    /// Probed live headroom for one Metal pre-load admission decision. Unlike
    /// planning headroom this deliberately reads the volatile live-available
    /// sample: a Metal model load allocates its full physical working set at
    /// the load fence, so starting one while the device has less free memory
    /// than the new model's static estimate turns an admission question into
    /// a command-buffer OOM. What the sample bounds follows
    /// [`MetalLoadAdmission`]: the GPU working set by default, the host's
    /// available memory too under `strict`, and nothing under `off` (`None`).
    /// Returns `None` when no trustworthy live sample exists (probe
    /// unavailable); callers must treat that as "fall back to planning
    /// headroom", never as unlimited. CPU keeps planning headroom only: host
    /// availability is compressed/swapped and must not gate loads.
    pub(crate) fn live_preload_headroom_bytes(&self, backend: BackendKind) -> Result<Option<u64>> {
        if backend != BackendKind::Metal || self.metal_load_admission() == MetalLoadAdmission::Off {
            return Ok(None);
        }
        self.live_headroom_in_domain(backend)
    }

    /// The provider's [`MetalLoadAdmission`] ([`MetalLoadAdmission::Advisory`]
    /// when it has none).
    pub(crate) fn metal_load_admission(&self) -> MetalLoadAdmission {
        self.provider.metal_load_admission().unwrap_or_default()
    }

    /// The host's available memory when it is below `required_bytes` and
    /// does not gate the load (Metal under [`MetalLoadAdmission::Advisory`]):
    /// macOS will page other processes to make room. `None` when the host
    /// has room, gates already, or reports nothing.
    pub(crate) fn advisory_host_shortfall(
        &self,
        backend: BackendKind,
        required_bytes: u64,
    ) -> Option<u64> {
        if backend != BackendKind::Metal
            || self.metal_load_admission() != MetalLoadAdmission::Advisory
        {
            return None;
        }
        self.provider
            .advisory_host_available_bytes()
            .filter(|available| *available < required_bytes)
    }

    /// Publish a post-release physical observation before another guarded
    /// reservation is admitted. The reservation ledger remains authoritative;
    /// this only prevents a cached provider from hiding newly freed memory.
    pub(crate) fn refresh_physical_capacity_after_release(&self) -> PhysicalCapacitySnapshot {
        let physical =
            self.normalized_physical_snapshot_from(self.provider.refresh_after_release());
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let capacity = match self.capacity_enforcement {
            CapacityEnforcement::Guarded => physical.capacity,
            CapacityEnforcement::Advisory => {
                advisory_ledger_capacity(self.provider.configured_budget(), self.domain_policy)
            }
        };
        state.ledger.update_capacity(capacity);
        physical
    }

    pub fn reserve(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
    ) -> Result<ResourceLease> {
        self.reserve_with_initial_materialized(owner, resources, ResourceVector::zero())
    }

    /// Reserve transient physical workspace for exactly one physical batch.
    /// Persistent KV/session state must use a cache lease instead, so a batch
    /// workspace can always be released when its dispatch transaction ends.
    pub(crate) fn reserve_batch_workspace(
        self: &Arc<Self>,
        execution_group: crate::engine::ExecutionGroupId,
        batch_id: crate::engine::BatchId,
        resources: ResourceVector,
    ) -> Result<BatchWorkspaceLease> {
        if !matches!(resources.kv_bytes, ResourceAmount::Known(0)) {
            return Err(Error::InvalidInput(
                "batch workspace cannot own persistent KV resources".to_string(),
            ));
        }
        if !matches!(resources.compute_slots, ResourceAmount::Known(0)) {
            return Err(Error::InvalidInput(
                "batch workspace cannot own execution permits".to_string(),
            ));
        }
        let owner = ReservationOwner::new(
            ReservationClass::BatchWorkspace,
            format!("{}:{}", execution_group.get(), batch_id.get()),
        );
        let lease = self.reserve(owner, resources)?;
        Ok(BatchWorkspaceLease {
            execution_group,
            batch_id,
            lease,
        })
    }

    /// Atomically establish immutable authorization for a resource claim whose
    /// initial physical allocation is already reflected by the provider.
    /// Guarded authorities charge live headroom only for future, unmaterialized
    /// growth; advisory authorities retain the same observation for accounting.
    pub fn reserve_with_initial_materialized(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
        materialized: ResourceVector,
    ) -> Result<ResourceLease> {
        self.reserve_internal(owner, resources, materialized)
    }

    fn reserve_internal(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
        materialized: ResourceVector,
    ) -> Result<ResourceLease> {
        let lease_resources = resources;
        let resources = normalize_resource_domains(resources, self.domain_policy)?;
        let materialized = normalize_resource_domains(materialized, self.domain_policy)?;
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(format!(
                "resource reservation for {} contains an unresolved quantity",
                owner.key
            )));
        }
        if !materialized.is_fully_known() {
            return Err(Error::InvalidInput(format!(
                "initial materialized usage for {} contains an unresolved quantity",
                owner.key
            )));
        }
        if !materialized.fits_within(resources) {
            return Err(Error::InvalidInput(format!(
                "initial materialized usage for {} exceeds its authorization",
                owner.key
            )));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        if let Some(reason) = state.poisoned.as_ref() {
            return Err(Error::InferenceError(format!(
                "backend resource authority is poisoned and must be recreated: {reason}"
            )));
        }
        let reservation = match self.capacity_enforcement {
            CapacityEnforcement::Guarded => {
                // Serialize the observation with ledger mutation. `available`
                // already excludes materialized allocations, but it cannot see
                // reservations that have not allocated yet. Charge every
                // pending claim against live headroom so concurrent guarded
                // reservations cannot spend it more than once.
                let physical = self.normalized_physical_snapshot();
                state.ledger.update_capacity(physical.capacity);
                let pending = resources.positive_growth_over(materialized)?;
                let existing_pending = state.pending_resources()?;
                let live_claim = existing_pending.checked_add(pending)?;
                if !live_claim.fits_within(physical.available) {
                    tracing::warn!(
                        "insufficient live physical capacity for {:?}: new_pending={pending:?}, existing_pending={existing_pending:?}, live_claim={live_claim:?}, physical_available={:?}, physical_capacity={:?}, live_by_class={:?}",
                        owner.class, physical.available, physical.capacity, state.pending_by_class(None)?
                    );
                    return Err(Error::Overloaded(live_capacity_reason(
                        pending,
                        existing_pending,
                        physical.available,
                    )));
                }
                state.ledger.reserve(resources)?
            }
            CapacityEnforcement::Advisory => {
                state.ledger.update_capacity(advisory_ledger_capacity(
                    self.provider.configured_budget(),
                    self.domain_policy,
                ));
                if owner.class == ReservationClass::Request {
                    // A request's working set must exist when it runs, so it
                    // is also checked against the operating system's live
                    // sample (Metal working set, CPU MemAvailable or cgroup),
                    // less every unallocated claim, not only the ledger:
                    // otherwise memory used elsewhere surfaces as a Metal OOM
                    // or swap in the middle of the request (ADM-4).
                    let physical = self.normalized_physical_snapshot();
                    if matches!(
                        physical.source,
                        CapacitySource::OperatingSystem | CapacitySource::MetalWorkingSet
                    ) {
                        let pending = resources.positive_growth_over(materialized)?;
                        let existing_pending = state.pending_resources()?;
                        let live_claim = existing_pending.checked_add(pending)?;
                        let live = physical.available;
                        let exceeds = |claim: ResourceAmount, available: ResourceAmount| {
                            matches!(
                                (claim, available),
                                (ResourceAmount::Known(claim), ResourceAmount::Known(available))
                                    if claim > available
                            )
                        };
                        if exceeds(live_claim.host_bytes, live.host_bytes)
                            || exceeds(live_claim.unified_bytes, live.unified_bytes)
                        {
                            tracing::warn!(
                                "insufficient live memory for a request: new_pending={pending:?}, existing_pending={existing_pending:?}, live_claim={live_claim:?}, physical_available={live:?}, live_by_class={:?}",
                                state.pending_by_class(None)?
                            );
                            return Err(Error::Overloaded(live_capacity_reason(
                                pending,
                                existing_pending,
                                live,
                            )));
                        }
                    }
                }
                state.ledger.reserve(resources)?
            }
        };
        state.owners.insert(reservation.id, owner);
        state.materialized.insert(reservation.id, materialized);
        Ok(ResourceLease {
            authority: self.clone(),
            id: Some(reservation.id),
            resources: lease_resources,
        })
    }

    fn release(&self, id: ReservationId) {
        if let Ok(mut state) = self.state.lock() {
            state.owners.remove(&id);
            state.materialized.remove(&id);
            let _ = state.ledger.release(id);
        }
    }

    fn resize(&self, id: ReservationId, resources: ResourceVector) -> Result<()> {
        let resources = normalize_resource_domains(resources, self.domain_policy)?;
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "resource resize contains an unresolved quantity".to_string(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        if let Some(reason) = state.poisoned.as_ref() {
            return Err(Error::InferenceError(format!(
                "backend resource authority is poisoned and must be recreated: {reason}"
            )));
        }
        let current = state.ledger.reservation(id).ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        let materialized = state
            .materialized
            .get(&id)
            .copied()
            .unwrap_or_else(ResourceVector::zero);
        if !materialized.fits_within(resources) {
            return Err(Error::InvalidInput(
                "resource resize would shrink authorization below materialized usage".to_string(),
            ));
        }
        let resized = if self.capacity_enforcement == CapacityEnforcement::Guarded {
            let current_pending = current.positive_growth_over(materialized)?;
            let next_pending = resources.positive_growth_over(materialized)?;
            let other_pending = state.pending_resources()?.checked_sub(current_pending)?;
            let live_claim = other_pending.checked_add(next_pending)?;
            let physical = self.normalized_physical_snapshot();
            state.ledger.update_capacity(physical.capacity);
            if !live_claim.fits_within(physical.available) {
                tracing::warn!(
                    "insufficient live physical capacity for resource lease growth: next_pending={next_pending:?}, other_pending={other_pending:?}, live_claim={live_claim:?}, physical_available={:?}, physical_capacity={:?}, other_live_by_class={:?}",
                    physical.available, physical.capacity, state.pending_by_class(Some(id))?
                );
                return Err(Error::Overloaded(live_capacity_reason(
                    next_pending,
                    other_pending,
                    physical.available,
                )));
            }
            state.ledger.resize(id, resources)?
        } else {
            state.ledger.update_capacity(advisory_ledger_capacity(
                self.provider.configured_budget(),
                self.domain_policy,
            ));
            state.ledger.resize(id, resources)?
        };
        if !resized {
            return Err(Error::InferenceError(
                "resource lease disappeared during resize".to_string(),
            ));
        }
        Ok(())
    }

    /// Return part of a reservation. It never consults live headroom or the
    /// ledger capacity, since it only gives capacity back, but it cannot
    /// drop below what the lease has materialized.
    fn shrink(&self, id: ReservationId, resources: ResourceVector) -> Result<()> {
        let resources = normalize_resource_domains(resources, self.domain_policy)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        let materialized = state
            .materialized
            .get(&id)
            .copied()
            .unwrap_or_else(ResourceVector::zero);
        if !materialized.fits_within(resources) {
            return Err(Error::InvalidInput(
                "resource shrink would drop authorization below materialized usage".to_string(),
            ));
        }
        if !state.ledger.shrink(id, resources)? {
            return Err(Error::InferenceError(
                "resource lease is no longer active".to_string(),
            ));
        }
        Ok(())
    }

    fn record_materialized_usage(
        &self,
        id: ReservationId,
        resources: ResourceVector,
    ) -> Result<()> {
        let resources = normalize_resource_domains(resources, self.domain_policy)?;
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "materialized resource usage contains an unresolved quantity".to_string(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        let reserved = state.ledger.reservation(id).ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        let current = state
            .materialized
            .get(&id)
            .copied()
            .unwrap_or_else(ResourceVector::zero);
        if !resources.fits_within(reserved) {
            return Err(state.materialization_exceeded(
                id,
                "record_materialized_usage",
                reserved,
                resources,
            ));
        }
        if !current.fits_within(resources) {
            // An owner's accounting bug, never the client's input.
            return Err(Error::InferenceError(
                "materialized resource usage cannot decrease through post-allocation observation; the owning runtime must restore its pending claim before freeing physical memory"
                    .to_string(),
            ));
        }
        state.materialized.insert(id, resources);
        Ok(())
    }

    fn prepare_materialized_release(
        &self,
        id: ReservationId,
        resources: ResourceVector,
    ) -> Result<()> {
        let resources = normalize_resource_domains(resources, self.domain_policy)?;
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "materialized resource release contains an unresolved quantity".to_string(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        let reserved = state.ledger.reservation(id).ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        let current = state
            .materialized
            .get(&id)
            .copied()
            .unwrap_or_else(ResourceVector::zero);
        if !resources.fits_within(reserved) {
            return Err(state.materialization_exceeded(
                id,
                "prepare_materialized_release",
                reserved,
                resources,
            ));
        }
        if !resources.fits_within(current) {
            return Err(Error::InvalidInput(
                "ordered materialized release cannot increase physical usage".to_string(),
            ));
        }

        // Convert the allocation back into a pending claim before making the
        // physical bytes visible as live headroom. A racing admission after
        // this transition sees the restored claim; an admission before it
        // still sees the allocation excluded from physical headroom.
        state.materialized.insert(id, resources);
        Ok(())
    }

    fn normalized_physical_snapshot(&self) -> PhysicalCapacitySnapshot {
        self.normalized_physical_snapshot_from(self.provider.snapshot())
    }

    fn normalized_physical_snapshot_from(
        &self,
        physical: PhysicalCapacitySnapshot,
    ) -> PhysicalCapacitySnapshot {
        PhysicalCapacitySnapshot {
            capacity: normalize_physical_resource_domains(physical.capacity, self.domain_policy),
            available: normalize_physical_resource_domains(physical.available, self.domain_policy),
            source: physical.source,
        }
    }
}

fn known_bytes(amount: ResourceAmount) -> u64 {
    match amount {
        ResourceAmount::Known(bytes) => bytes,
        ResourceAmount::Unknown => 0,
    }
}

/// The short reason a client sees when `subject` needs more memory than is
/// available: "{subject}: needs X GB, Y GB available". Callers log the
/// detail (headroom, reserve, geometry) beside it.
pub(crate) fn capacity_shortfall_reason(
    subject: impl std::fmt::Display,
    need_bytes: u64,
    available_bytes: u64,
) -> String {
    format!(
        "{subject}: needs {:.2} GB, {:.2} GB available",
        need_bytes as f64 / 1e9,
        available_bytes as f64 / 1e9
    )
}

/// The short reason a client sees when load-time state does not fit beside
/// the safety reserve its fit keeps free: "{subject}: needs X plus a Y
/// safety reserve, Z available", where Z is the planning headroom the
/// reserve comes out of. Without the reserve, a small state against a
/// headroom below it read "needs 0.00 GB, 0.00 GB available".
pub(crate) fn state_shortfall_reason(
    subject: impl std::fmt::Display,
    need_bytes: u64,
    headroom_bytes: u64,
    safety_reserve_bytes: u64,
) -> String {
    format!(
        "{subject}: needs {} plus a {} safety reserve, {} available",
        display_bytes(need_bytes),
        display_bytes(safety_reserve_bytes),
        display_bytes(headroom_bytes)
    )
}

/// Bytes in GB to two places, or in MB below 0.1 GB so small figures do
/// not round to zero.
fn display_bytes(bytes: u64) -> String {
    if bytes >= 100_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.1} MB", bytes as f64 / 1e6)
    }
}

/// Whether `error` refuses memory that admitted work holds now and gives
/// back when it finishes: the live-memory check or the reservation ledger.
/// A request too large for the machine, a full job queue or a draining
/// runtime is not such a refusal.
pub(crate) fn is_held_capacity_refusal(error: &Error) -> bool {
    matches!(
        error,
        Error::Overloaded(message)
            if message.starts_with("insufficient live")
                || message == "requested resources exceed available capacity"
    )
}

/// The short reason a client sees when live memory refuses a claim: the
/// first domain where `need` plus what is already reserved but not yet
/// allocated (admitted requests and loaded models' pending claims)
/// exceeds `available`, in GB. The full vectors are logged by the caller.
fn live_capacity_reason(
    need: ResourceVector,
    held: ResourceVector,
    available: ResourceVector,
) -> String {
    let domains: [(&str, fn(ResourceVector) -> ResourceAmount); 6] = [
        ("memory", |vector| vector.host_bytes),
        ("memory", |vector| vector.unified_bytes),
        ("GPU memory", |vector| vector.device_bytes),
        ("KV memory", |vector| vector.kv_bytes),
        ("workspace memory", |vector| vector.temporary_bytes),
        ("compute slots", |vector| vector.compute_slots),
    ];
    for (label, amount) in domains {
        let (ResourceAmount::Known(need), ResourceAmount::Known(held)) =
            (amount(need), amount(held))
        else {
            continue;
        };
        let claim = need.saturating_add(held);
        if ResourceAmount::Known(claim).fits(amount(available)) {
            continue;
        }
        let ResourceAmount::Known(available) = amount(available) else {
            return format!("insufficient live {label}: its capacity is unknown");
        };
        let free = available.saturating_sub(held);
        if label == "compute slots" {
            return format!("insufficient {label}: needs {need}, {free} available");
        }
        let gb = |bytes: u64| bytes as f64 / 1e9;
        return format!(
            "insufficient live {label}: needs {:.2} GB, {:.2} GB available ({:.2} GB reserved, not yet allocated)",
            gb(need),
            gb(free),
            gb(held)
        );
    }
    "insufficient live memory".to_string()
}

fn advisory_ledger_capacity(
    configured_budget: Option<ResourceVector>,
    domain_policy: ResourceDomainPolicy,
) -> ResourceVector {
    if let Some(budget) = configured_budget {
        return normalize_physical_resource_domains(budget, domain_policy);
    }
    ResourceVector {
        host_bytes: ResourceAmount::Known(if domain_policy == ResourceDomainPolicy::Independent {
            u64::MAX
        } else {
            0
        }),
        device_bytes: ResourceAmount::Known(u64::MAX),
        unified_bytes: ResourceAmount::Known(u64::MAX),
        kv_bytes: ResourceAmount::Known(u64::MAX),
        temporary_bytes: ResourceAmount::Known(u64::MAX),
        compute_slots: ResourceAmount::Known(u64::MAX),
    }
}

fn normalize_resource_domains(
    mut resources: ResourceVector,
    policy: ResourceDomainPolicy,
) -> Result<ResourceVector> {
    if policy == ResourceDomainPolicy::SharedHostUnified {
        resources.unified_bytes = resources.host_bytes.checked_add(resources.unified_bytes)?;
        resources.host_bytes = ResourceAmount::Known(0);
    }
    Ok(resources)
}

fn normalize_physical_resource_domains(
    mut resources: ResourceVector,
    policy: ResourceDomainPolicy,
) -> ResourceVector {
    match normalize_resource_domains(resources, policy) {
        Ok(resources) => resources,
        Err(_) => {
            // A provider overflow is untrustworthy capacity, never infinite
            // capacity and never a process panic. Preserve unrelated domains
            // while making the aliased pool fail closed.
            resources.host_bytes = ResourceAmount::Known(0);
            resources.unified_bytes = ResourceAmount::Unknown;
            resources
        }
    }
}

#[derive(Debug)]
pub struct ResourceLease {
    authority: Arc<ResourceAuthority>,
    id: Option<ReservationId>,
    resources: ResourceVector,
}

/// Short-lived resource authorization scoped to one physical batch dispatch.
/// Dropping this value releases the workspace reservation; it must never be
/// stored in per-session state or an [`crate::engine::ExecutionPlan`].
#[derive(Debug)]
pub struct BatchWorkspaceLease {
    execution_group: crate::engine::ExecutionGroupId,
    batch_id: crate::engine::BatchId,
    lease: ResourceLease,
}

impl BatchWorkspaceLease {
    pub fn execution_group(&self) -> crate::engine::ExecutionGroupId {
        self.execution_group
    }

    pub fn batch_id(&self) -> crate::engine::BatchId {
        self.batch_id
    }

    pub fn resources(&self) -> ResourceVector {
        self.lease.resources()
    }

    pub fn record_materialized_usage(&self, resources: ResourceVector) -> Result<()> {
        self.lease.record_materialized_usage(resources)
    }
}

impl ResourceLease {
    pub fn resources(&self) -> ResourceVector {
        self.resources
    }

    /// Resize before additional physical allocation. Only positive growth is
    /// compared with current live headroom; the ledger still validates the
    /// complete replacement against total capacity.
    pub fn resize(&mut self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.resize(id, resources)?;
        self.resources = resources;
        Ok(())
    }

    /// Give back part of the authorization once it can no longer be used.
    /// Unlike [`Self::resize`] it never consults live headroom, so it
    /// succeeds whenever the materialized usage still fits.
    pub fn shrink(&mut self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.shrink(id, resources)?;
        self.resources = resources;
        Ok(())
    }

    /// Record a physical allocation that was just observed without changing
    /// the authorization established before allocation. Observed usage must
    /// fit within that authorization; callers must use `resize` before any
    /// physical growth. Observations are monotonic for the lease lifetime.
    pub fn reconcile_materialized(&self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.record_materialized_usage(id, resources)
    }

    /// Record the portion of this reservation that is physically allocated
    /// without relinquishing any of the capacity authorized for future growth.
    /// This public post-allocation observation is monotonic. The internal
    /// resource owner restores pending claims before releasing allocations.
    pub fn record_materialized_usage(&self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.record_materialized_usage(id, resources)
    }

    /// Move physical usage back into the lease's pending claim. Callers must
    /// complete this transition before dropping or replacing the corresponding
    /// allocation. It can temporarily overcharge physical capacity, but it
    /// cannot expose released headroom without its retained future claim.
    pub(crate) fn prepare_materialized_release(&self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.prepare_materialized_release(id, resources)
    }
}

impl Drop for ResourceLease {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            self.authority.release(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug)]
    struct TestProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for TestProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }
    }

    #[derive(Debug)]
    struct LiveProvider {
        capacity: u64,
        available: AtomicU64,
    }

    #[derive(Debug)]
    struct BudgetProvider {
        snapshot: PhysicalCapacitySnapshot,
        budget: ResourceVector,
    }

    impl PhysicalCapacityProvider for BudgetProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }

        fn configured_budget(&self) -> Option<ResourceVector> {
            Some(self.budget)
        }
    }

    #[derive(Debug)]
    struct ReleaseAwareProvider {
        capacity: u64,
        cached_available: AtomicU64,
        released_available: AtomicU64,
        refreshes: AtomicU64,
    }

    impl PhysicalCapacityProvider for ReleaseAwareProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: slots(self.capacity),
                available: slots(self.cached_available.load(Ordering::Acquire)),
                source: CapacitySource::Test,
            }
        }

        fn refresh_after_release(&self) -> PhysicalCapacitySnapshot {
            self.refreshes.fetch_add(1, Ordering::AcqRel);
            self.cached_available.store(
                self.released_available.load(Ordering::Acquire),
                Ordering::Release,
            );
            self.snapshot()
        }
    }

    impl LiveProvider {
        fn set_available(&self, available: u64) {
            self.available.store(available, Ordering::Release);
        }
    }

    impl PhysicalCapacityProvider for LiveProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: slots(self.capacity),
                available: slots(self.available.load(Ordering::Acquire)),
                source: CapacitySource::Test,
            }
        }
    }

    fn slots(value: u64) -> ResourceVector {
        ResourceVector {
            compute_slots: ResourceAmount::Known(value),
            ..ResourceVector::zero()
        }
    }

    fn host_bytes(value: u64) -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(value),
            ..ResourceVector::zero()
        }
    }

    fn device_bytes(value: u64) -> ResourceVector {
        ResourceVector {
            device_bytes: ResourceAmount::Known(value),
            ..ResourceVector::zero()
        }
    }

    #[test]
    fn reservation_is_transactional_and_releases_exactly_once() {
        let mut ledger = ResourceLedger::new(slots(2));
        let first = ledger.reserve(slots(2)).unwrap();
        assert!(ledger.reserve(slots(1)).is_err());
        assert_eq!(ledger.used(), slots(2));
        assert!(ledger.release(first.id).unwrap());
        assert!(!ledger.release(first.id).unwrap());
        assert_eq!(ledger.used(), slots(0));
    }

    #[test]
    fn unknown_capacity_is_not_treated_as_infinite() {
        let mut ledger = ResourceLedger::new(ResourceVector::default());
        assert!(ledger.reserve(slots(1)).is_err());
        assert_eq!(ledger.used(), ResourceVector::zero());
    }

    #[test]
    fn unresolved_reservation_is_rejected_without_poisoning_usage() {
        let mut ledger = ResourceLedger::new(slots(2));
        assert!(matches!(
            ledger.reserve(ResourceVector::default()),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(ledger.used(), ResourceVector::zero());
    }

    #[test]
    fn shared_authority_serializes_different_owner_classes() {
        let provider = Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: slots(2),
                available: slots(2),
                source: CapacitySource::Test,
            },
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "model"),
                slots(1),
            )
            .unwrap();
        let request = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "request"),
                slots(1),
            )
            .unwrap();
        assert!(authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "cache"),
                slots(1),
            )
            .is_err());
        assert_eq!(authority.snapshot().reserved, slots(2));
        drop((model, request));
        assert_eq!(authority.snapshot().reserved, slots(0));
    }

    #[test]
    fn poisoned_authority_rejects_new_work_but_still_releases_existing_leases() {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: slots(2),
                available: slots(2),
                source: CapacitySource::Test,
            },
        })));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "loaded"),
                slots(1),
            )
            .unwrap();
        authority.poison("Metal command buffer OOM");
        assert_eq!(
            authority.poison_reason().as_deref(),
            Some("Metal command buffer OOM")
        );
        let error = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "new"),
                slots(1),
            )
            .unwrap_err();
        assert!(error.to_string().contains("must be recreated"));
        drop(lease);
        assert_eq!(authority.snapshot().reserved, slots(0));
    }

    #[test]
    fn post_release_refresh_makes_freed_capacity_visible_to_next_admission() {
        let provider = Arc::new(ReleaseAwareProvider {
            capacity: 2,
            cached_available: AtomicU64::new(0),
            released_available: AtomicU64::new(2),
            refreshes: AtomicU64::new(0),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));

        assert!(authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "before-release"),
                slots(1),
            )
            .is_err());
        let refreshed = authority.refresh_physical_capacity_after_release();
        assert_eq!(refreshed.available, slots(2));
        assert_eq!(provider.refreshes.load(Ordering::Acquire), 1);

        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "after-release"),
                slots(1),
            )
            .unwrap();
        drop(lease);
    }

    #[test]
    fn batch_workspace_is_exactly_scoped_and_released_on_drop() {
        let provider = Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: host_bytes(16),
                available: host_bytes(16),
                source: CapacitySource::Test,
            },
        });
        let authority = Arc::new(ResourceAuthority::new(provider));

        let workspace = authority
            .reserve_batch_workspace(
                crate::engine::ExecutionGroupId::new(3),
                crate::engine::BatchId::new(9),
                host_bytes(8),
            )
            .expect("batch workspace");

        assert_eq!(workspace.execution_group().get(), 3);
        assert_eq!(workspace.batch_id().get(), 9);
        assert_eq!(workspace.resources(), host_bytes(8));
        assert_eq!(authority.snapshot().reservations, 1);
        drop(workspace);
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[test]
    fn batch_workspace_cannot_hide_persistent_state_or_execution_permits() {
        let provider = Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: host_bytes(16),
                available: host_bytes(16),
                source: CapacitySource::Test,
            },
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let mut persistent = host_bytes(4);
        persistent.kv_bytes = ResourceAmount::Known(1);
        let mut permit = host_bytes(4);
        permit.compute_slots = ResourceAmount::Known(1);

        assert!(matches!(
            authority.reserve_batch_workspace(
                crate::engine::ExecutionGroupId::new(1),
                crate::engine::BatchId::new(1),
                persistent,
            ),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            authority.reserve_batch_workspace(
                crate::engine::ExecutionGroupId::new(1),
                crate::engine::BatchId::new(2),
                permit,
            ),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(authority.snapshot().reservations, 0);
    }

    /// The live Kokoro refusal sent the client every resource vector and the
    /// per-class pending table; it now reads like this.
    #[test]
    fn state_shortfall_reason_names_the_safety_reserve() {
        // The refusal that used to read "needs 0.00 GB, 0.00 GB available".
        const RESERVE: u64 = 1_073_741_824;
        assert_eq!(
            state_shortfall_reason("X's working state", 3_200_000, 900_000_000, RESERVE),
            "X's working state: needs 3.2 MB plus a 1.07 GB safety reserve, 0.90 GB available"
        );
        assert_eq!(
            state_shortfall_reason("minimum", 500_654_080, 1_342_615_072, RESERVE),
            "minimum: needs 0.50 GB plus a 1.07 GB safety reserve, 1.34 GB available"
        );
    }

    #[test]
    fn live_capacity_reason_is_short_and_in_gb() {
        let unified = |bytes| ResourceVector {
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        };
        assert_eq!(
            live_capacity_reason(
                unified(3_337_390_021),
                unified(104_000_000),
                unified(2_304_168_748)
            ),
            "insufficient live memory: needs 3.34 GB, 2.20 GB available (0.10 GB reserved, not yet allocated)"
        );
    }

    #[test]
    fn live_capacity_failure_is_transactional() {
        let provider = Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: slots(2),
                available: slots(0),
                source: CapacitySource::Test,
            },
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let error = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "request"),
                slots(1),
            )
            .unwrap_err();
        // The client gets a short reason; the vectors go to the log.
        assert_eq!(
            error.to_string(),
            "Inference capacity unavailable: insufficient compute slots: needs 1, 0 available"
        );
        assert_eq!(authority.snapshot().reserved, slots(0));
    }

    #[test]
    fn pending_class_diagnostics_are_bounded_and_exclude_materialized_allocations() {
        let provider = Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: slots(1000),
                available: slots(500),
                source: CapacitySource::Test,
            },
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let cache = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Cache, "private-cache-key"),
                slots(100),
                slots(80),
            )
            .unwrap();
        let requests: Vec<_> = (0..100)
            .map(|_| {
                authority
                    .reserve(
                        ReservationOwner::new(ReservationClass::Request, "private-request-key"),
                        slots(1),
                    )
                    .unwrap()
            })
            .collect();
        {
            let state = authority.state.lock().unwrap();
            let summaries = state.pending_by_class(None).unwrap();
            assert_eq!(summaries.len(), 5);
            assert_eq!(summaries[1].reservations, 100);
            assert_eq!(summaries[1].pending, slots(100));
            assert_eq!(summaries[2].reservations, 1);
            assert_eq!(summaries[2].pending, slots(20));
            // The per-class summary names which residency holds the device:
            // the cache lease reserved 100 slots, materialized 80, and still
            // owes 20 pending.
            assert_eq!(summaries[2].reserved, slots(100));
            assert_eq!(summaries[2].materialized, slots(80));
            let total = summaries
                .iter()
                .fold(ResourceVector::zero(), |sum, summary| {
                    sum.checked_add(summary.pending).unwrap()
                });
            assert_eq!(total, state.pending_resources().unwrap());
            let excluded = state.pending_by_class(cache.id).unwrap();
            assert_eq!(excluded[2].reservations, 0);
            assert_eq!(excluded[2].pending, slots(0));
            assert_eq!(excluded[2].reserved, slots(0));
        }
        let before = authority.snapshot();
        let message = authority
            .reserve(
                ReservationOwner::new(ReservationClass::BatchWorkspace, "private-incoming-key"),
                slots(400),
            )
            .unwrap_err()
            .to_string();
        // The per-class table is logged; the client gets one short line.
        assert!(!message.contains("live_by_class="));
        assert!(!message.contains("private-"));
        assert!(message.len() < 160);
        assert_eq!(authority.snapshot(), before);
        let growth_message = authority
            .resize(cache.id.unwrap(), slots(600))
            .unwrap_err()
            .to_string();
        assert!(!growth_message.contains("other_live_by_class="));
        assert!(!growth_message.contains("private-"));
        assert!(growth_message.len() < 160);
        assert_eq!(authority.snapshot(), before);
        drop((cache, requests));
    }

    /// A shrink only gives capacity back: it succeeds when a guarded resize
    /// to the same size is refused on live headroom, it never grows a lease
    /// and it never drops below what the lease has materialized.
    #[test]
    fn lease_shrink_returns_capacity_without_a_live_check() {
        let provider = Arc::new(LiveProvider {
            capacity: 10,
            available: AtomicU64::new(10),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let mut job = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "job"),
                slots(6),
            )
            .unwrap();
        job.record_materialized_usage(slots(2)).unwrap();
        let _other = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "other"),
                slots(3),
            )
            .unwrap();
        provider.set_available(1);

        assert!(matches!(job.resize(slots(4)), Err(Error::Overloaded(_))));
        job.shrink(slots(4)).unwrap();
        assert_eq!(job.resources(), slots(4));
        assert_eq!(authority.snapshot().reserved, slots(7));
        assert!(matches!(job.shrink(slots(5)), Err(Error::InvalidInput(_))));
        assert!(matches!(job.shrink(slots(1)), Err(Error::InvalidInput(_))));
        assert_eq!(authority.snapshot().reserved, slots(7));
        drop(job);
        assert_eq!(authority.snapshot().reserved, slots(3));
    }

    #[test]
    fn advisory_authority_tracks_every_owner_class_without_physical_rejection() {
        let provider = Arc::new(LiveProvider {
            capacity: 2,
            available: AtomicU64::new(0),
        });
        let authority = Arc::new(ResourceAuthority::new_advisory(provider));
        let model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "model"),
                host_bytes(3),
            )
            .unwrap();
        let request = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Request, "request"),
                host_bytes(2),
                host_bytes(1),
            )
            .unwrap();
        let cache = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "cache"),
                host_bytes(4),
            )
            .unwrap();
        let pipeline = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Pipeline, "pipeline"),
                host_bytes(5),
            )
            .unwrap();
        let workspace = authority
            .reserve_batch_workspace(
                crate::engine::ExecutionGroupId::new(1),
                crate::engine::BatchId::new(1),
                host_bytes(6),
            )
            .unwrap();

        assert_eq!(authority.snapshot().reserved, host_bytes(20));
        assert_eq!(authority.snapshot().reservations, 5);
        model.reconcile_materialized(host_bytes(3)).unwrap();
        request.reconcile_materialized(host_bytes(2)).unwrap();

        drop((workspace, pipeline, cache, request, model));
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
        assert_eq!(authority.snapshot().reservations, 0);
    }

    #[test]
    fn portable_planning_uses_stable_total_not_volatile_live_available() {
        let capacity = host_bytes(100);
        let authority = Arc::new(ResourceAuthority::new_advisory(Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: host_bytes(0),
                source: CapacitySource::Test,
            },
        })));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "resident"),
                host_bytes(30),
            )
            .unwrap();
        assert_eq!(authority.planning_headroom().unwrap(), host_bytes(70));
        drop(lease);
        assert_eq!(authority.planning_headroom().unwrap(), host_bytes(100));
    }

    #[test]
    fn guarded_cuda_planning_intersects_ledger_and_live_device_headroom() {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: device_bytes(100),
                available: device_bytes(25),
                source: CapacitySource::Test,
            },
        })));
        let _pending = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "pending"),
                device_bytes(5),
            )
            .unwrap();

        assert_eq!(authority.planning_headroom().unwrap(), device_bytes(95));
        assert_eq!(
            authority
                .planning_headroom_bytes(BackendKind::Cuda)
                .unwrap(),
            ResourceAmount::Known(20)
        );
    }

    #[test]
    fn guarded_cuda_planning_preserves_a_lower_operator_budget() {
        let authority = ResourceAuthority::new(Arc::new(BudgetProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: device_bytes(100),
                available: device_bytes(25),
                source: CapacitySource::Test,
            },
            budget: device_bytes(15),
        }));

        assert_eq!(
            authority
                .planning_headroom_bytes(BackendKind::Cuda)
                .unwrap(),
            ResourceAmount::Known(15)
        );
    }

    fn unified_bytes(value: u64) -> ResourceVector {
        ResourceVector {
            unified_bytes: ResourceAmount::Known(value),
            ..ResourceVector::zero()
        }
    }

    #[derive(Debug)]
    struct UnifiedProvider {
        capacity: u64,
        available: u64,
        source: CapacitySource,
    }

    impl PhysicalCapacityProvider for UnifiedProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: unified_bytes(self.capacity),
                available: unified_bytes(self.available),
                source: self.source,
            }
        }
    }

    /// A Metal provider under one `IZWI_METAL_LOAD_ADMISSION` mode, with a
    /// fixed working-set headroom and host-available sample.
    #[derive(Debug)]
    struct MetalModeProvider {
        working_set_available: u64,
        host_available: u64,
        admission: MetalLoadAdmission,
    }

    impl PhysicalCapacityProvider for MetalModeProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let available = match self.admission {
                MetalLoadAdmission::Strict => self.working_set_available.min(self.host_available),
                MetalLoadAdmission::Advisory | MetalLoadAdmission::Off => {
                    self.working_set_available
                }
            };
            PhysicalCapacitySnapshot {
                capacity: unified_bytes(100),
                available: unified_bytes(available),
                source: CapacitySource::MetalWorkingSet,
            }
        }

        fn metal_load_admission(&self) -> Option<MetalLoadAdmission> {
            Some(self.admission)
        }

        fn advisory_host_available_bytes(&self) -> Option<u64> {
            (self.admission != MetalLoadAdmission::Strict).then_some(self.host_available)
        }
    }

    fn metal_mode_authority(admission: MetalLoadAdmission) -> Arc<ResourceAuthority> {
        Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(MetalModeProvider {
                working_set_available: 80,
                host_available: 20,
                admission,
            }),
        ))
    }

    #[test]
    fn metal_load_admission_parses_its_three_modes_and_defaults_to_advisory() {
        assert_eq!(MetalLoadAdmission::default(), MetalLoadAdmission::Advisory);
        for mode in [
            MetalLoadAdmission::Strict,
            MetalLoadAdmission::Advisory,
            MetalLoadAdmission::Off,
        ] {
            assert_eq!(MetalLoadAdmission::parse(mode.as_str()), Some(mode));
        }
        assert_eq!(
            MetalLoadAdmission::parse(" STRICT "),
            Some(MetalLoadAdmission::Strict)
        );
        assert_eq!(MetalLoadAdmission::parse("permissive"), None);
    }

    #[test]
    fn advisory_metal_preload_is_bounded_by_the_working_set_not_host_memory() {
        let authority = metal_mode_authority(MetalLoadAdmission::Advisory);
        // The host reports 20 available; the working set 80. A 60-byte load
        // is admitted and its host shortfall reported, not refused.
        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            Some(80)
        );
        assert_eq!(
            authority.advisory_host_shortfall(BackendKind::Metal, 60),
            Some(20)
        );
        assert_eq!(
            authority.advisory_host_shortfall(BackendKind::Metal, 15),
            None
        );
        assert_eq!(
            authority.advisory_host_shortfall(BackendKind::Cpu, 60),
            None
        );
    }

    #[test]
    fn strict_metal_preload_is_bounded_by_host_memory_too() {
        let authority = metal_mode_authority(MetalLoadAdmission::Strict);
        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            Some(20)
        );
        // Host memory gates already, so there is nothing to warn about.
        assert_eq!(
            authority.advisory_host_shortfall(BackendKind::Metal, 60),
            None
        );
    }

    #[test]
    fn off_skips_the_live_metal_preload_check() {
        let authority = metal_mode_authority(MetalLoadAdmission::Off);
        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            None
        );
        assert_eq!(
            authority.advisory_host_shortfall(BackendKind::Metal, 60),
            None
        );
    }

    #[test]
    fn advisory_metal_requests_are_not_refused_for_host_memory() {
        // ADM-4 checks a request against the live sample, which under the
        // advisory policy is the working set: 50 fits in 80 although the host
        // reports 20.
        let authority = metal_mode_authority(MetalLoadAdmission::Advisory);
        let request = authority.reserve(
            ReservationOwner::new(ReservationClass::Request, "request"),
            unified_bytes(50),
        );
        assert!(request.is_ok(), "{request:?}");
        let strict = metal_mode_authority(MetalLoadAdmission::Strict).reserve(
            ReservationOwner::new(ReservationClass::Request, "request"),
            unified_bytes(50),
        );
        assert!(matches!(strict, Err(Error::Overloaded(_))), "{strict:?}");
    }

    #[test]
    fn metal_live_preload_headroom_binds_when_allocations_exceed_estimates() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(UnifiedProvider {
                capacity: 100,
                available: 50,
                source: CapacitySource::Test,
            }),
        ));
        // The resident model reserved 30 and fully materialized, but the live
        // probe lost another 20 to allocations no static estimate tracks
        // (GGUF conversion scratch, pooled buffers): 100 − 30 − 20 = 50 free.
        let lease = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Model, "resident"),
                unified_bytes(30),
                unified_bytes(30),
            )
            .unwrap();

        // Static planning headroom trusts the ledger: 100 − 30 = 70. The
        // device reports only 50 free bytes, so the live sample must bind.
        assert_eq!(
            authority
                .planning_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            ResourceAmount::Known(70)
        );
        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            Some(50)
        );
        drop(lease);
    }

    #[test]
    fn metal_live_preload_headroom_excludes_pending_claims() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(UnifiedProvider {
                capacity: 100,
                available: 90,
                source: CapacitySource::Test,
            }),
        ));
        // 30 reserved, 20 already materialized: their bytes are inside the
        // provider's live `available` observation, so only the remaining 10
        // of pending claim is subtracted: 90 − 10 = 80.
        let lease = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Model, "resident"),
                unified_bytes(30),
                unified_bytes(20),
            )
            .unwrap();

        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            Some(80)
        );
        drop(lease);
    }

    #[test]
    fn asr_state_fit_headroom_takes_live_memory_below_the_ledger() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(UnifiedProvider {
                capacity: 100,
                available: 40,
                source: CapacitySource::Test,
            }),
        ));
        let _model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "model"),
                unified_bytes(30),
            )
            .unwrap();
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            // Planning: 100 - 30 reserved. Live: 40 available - 30 pending.
            assert_eq!(
                authority.planning_headroom_bytes(backend).unwrap(),
                ResourceAmount::Known(70)
            );
            assert_eq!(
                authority.state_fit_headroom_bytes(backend).unwrap(),
                ResourceAmount::Known(10)
            );
        }

        // Without a trustworthy live sample the ledger figure stands.
        let unprobed = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(UnifiedProvider {
                capacity: 100,
                available: 0,
                source: CapacitySource::Unavailable,
            }),
        ));
        assert_eq!(
            unprobed.state_fit_headroom_bytes(BackendKind::Metal).unwrap(),
            ResourceAmount::Known(100)
        );
        authority.poison("simulated backend failure");
        assert!(authority.state_fit_headroom_bytes(BackendKind::Cpu).is_err());
    }

    /// Advisory authorities still check a request against the operating
    /// system's live sample (less every unallocated claim); model and state
    /// reservations, and test-source samples, keep the ledger-only contract.
    #[test]
    fn advisory_request_reservations_check_live_memory() {
        let live = |source| {
            Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
                Arc::new(UnifiedProvider {
                    capacity: 1_000,
                    available: 100,
                    source,
                }),
            ))
        };
        for source in [
            CapacitySource::OperatingSystem,
            CapacitySource::MetalWorkingSet,
        ] {
            let authority = live(source);
            let request = |bytes| {
                authority.reserve(
                    ReservationOwner::new(ReservationClass::Request, "request"),
                    unified_bytes(bytes),
                )
            };
            assert!(matches!(request(150), Err(Error::Overloaded(_))));
            let first = request(60).unwrap();
            // 60 pending + 50 > 100 live.
            assert!(matches!(request(50), Err(Error::Overloaded(_))));
            // Bytes already allocated (the decoded input) are not pending.
            let observed = authority
                .reserve_with_initial_materialized(
                    ReservationOwner::new(ReservationClass::Request, "observed"),
                    unified_bytes(90),
                    unified_bytes(60),
                )
                .unwrap();
            drop((first, observed));
            let model = authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "model"),
                unified_bytes(500),
            );
            assert!(model.is_ok(), "{source:?}");
        }
        let test_source = live(CapacitySource::Test);
        assert!(test_source
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "request"),
                unified_bytes(150),
            )
            .is_ok());
    }

    #[derive(Debug)]
    struct ObservedProvider {
        observed: Option<u64>,
    }

    impl PhysicalCapacityProvider for ObservedProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: unified_bytes(10_000),
                available: unified_bytes(10_000),
                source: CapacitySource::Test,
            }
        }

        fn observed_device_usage_bytes(&self) -> Option<u64> {
            self.observed
        }
    }

    /// The untracked gauge is the observed device usage less the ledger's
    /// materialized bytes, and the one ASR safety reserve is the larger of it
    /// and the configured floor; without an observer the floor stands.
    #[test]
    fn untracked_device_memory_sizes_the_state_safety_reserve() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(ObservedProvider {
                observed: Some(900),
            }),
        ));
        let model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "model"),
                unified_bytes(600),
            )
            .unwrap();
        // Reserved but not yet allocated: everything observed is untracked.
        assert_eq!(
            authority.untracked_device_bytes(BackendKind::Metal),
            Some(900)
        );
        model.reconcile_materialized(unified_bytes(500)).unwrap();
        assert_eq!(
            authority.untracked_device_bytes(BackendKind::Metal),
            Some(400)
        );
        assert_eq!(authority.reserved_bytes_in_domain(BackendKind::Metal), 600);
        assert_eq!(
            authority.state_safety_reserve_bytes(BackendKind::Metal, 100),
            400
        );
        assert_eq!(
            authority.state_safety_reserve_bytes(BackendKind::Metal, 1_000),
            1_000
        );

        let unobserved = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(ObservedProvider { observed: None }),
        ));
        assert_eq!(unobserved.untracked_device_bytes(BackendKind::Cpu), None);
        assert_eq!(
            unobserved.state_safety_reserve_bytes(BackendKind::Cpu, 1_000),
            1_000
        );
    }

    #[test]
    fn live_preload_headroom_skips_cpu_and_unavailable_probes() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(UnifiedProvider {
                capacity: 100,
                available: 50,
                source: CapacitySource::Unavailable,
            }),
        ));

        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Cpu)
                .unwrap(),
            None
        );
        assert_eq!(
            authority
                .live_preload_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            None
        );
    }

    #[test]
    fn live_preload_headroom_fails_closed_when_poisoned() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(UnifiedProvider {
                capacity: 100,
                available: 50,
                source: CapacitySource::Test,
            }),
        ));
        authority.poison("simulated backend failure");

        assert!(authority
            .live_preload_headroom_bytes(BackendKind::Metal)
            .is_err());
    }

    #[test]
    fn shared_host_unified_planning_uses_one_canonical_domain_for_cpu_and_metal() {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(TestProvider {
                snapshot: PhysicalCapacitySnapshot {
                    capacity: host_bytes(100),
                    available: ResourceVector::zero(),
                    source: CapacitySource::Test,
                },
            }),
        ));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "resident"),
                host_bytes(30),
            )
            .unwrap();

        assert_eq!(
            authority.planning_headroom_bytes(BackendKind::Cpu).unwrap(),
            ResourceAmount::Known(70)
        );
        assert_eq!(
            authority
                .planning_headroom_bytes(BackendKind::Metal)
                .unwrap(),
            ResourceAmount::Known(70)
        );
        drop(lease);
    }

    #[test]
    fn advisory_resize_and_release_remain_transactional() {
        let provider = Arc::new(LiveProvider {
            capacity: 1,
            available: AtomicU64::new(0),
        });
        let authority = Arc::new(ResourceAuthority::new_advisory(provider));
        authority.refresh_physical_capacity_after_release();
        let mut lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "request"),
                host_bytes(3),
            )
            .unwrap();

        lease.resize(host_bytes(4)).unwrap();
        assert_eq!(authority.snapshot().reserved, host_bytes(4));
        drop(lease);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[test]
    fn advisory_physical_sampling_preserves_explicit_budget_ceiling() {
        let provider = Arc::new(BudgetProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: host_bytes(100),
                available: host_bytes(0),
                source: CapacitySource::Test,
            },
            budget: host_bytes(5),
        });
        let authority = Arc::new(ResourceAuthority::new_advisory(provider));
        authority.refresh_physical_capacity_after_release();
        let mut lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "within-budget"),
                host_bytes(4),
            )
            .expect("zero live availability remains advisory below the explicit budget");

        assert!(matches!(
            lease.resize(host_bytes(6)),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(authority.snapshot().reserved, host_bytes(4));
        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "over-budget"),
                host_bytes(2),
            ),
            Err(Error::Overloaded(_))
        ));
        drop(lease);
        authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "after-release"),
                host_bytes(5),
            )
            .expect("release restores the configured budget");
    }

    #[test]
    fn materialized_reservation_is_not_counted_twice_against_live_headroom() {
        let provider = Arc::new(LiveProvider {
            capacity: 10,
            available: AtomicU64::new(10),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "model"),
                slots(6),
            )
            .unwrap();

        // Simulate the model allocation becoming visible to the provider. The
        // six-unit model lease is already reflected in the four live units.
        provider.set_available(4);
        model.reconcile_materialized(slots(6)).unwrap();
        let request = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "request"),
                slots(1),
            )
            .unwrap();

        assert_eq!(authority.snapshot().reserved, slots(7));
        drop((request, model));
        assert_eq!(authority.snapshot().reserved, slots(0));
    }

    #[test]
    fn unmaterialized_reservations_remain_bounded_by_total_capacity() {
        let provider = Arc::new(LiveProvider {
            capacity: 10,
            available: AtomicU64::new(10),
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let _first = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "first"),
                slots(6),
            )
            .unwrap();

        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "second"),
                slots(5),
            ),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(authority.snapshot().reserved, slots(6));
    }

    #[test]
    fn pending_reservations_cannot_double_spend_external_headroom() {
        let vectors: [fn(u64) -> ResourceVector; 3] = [
            |value| ResourceVector {
                host_bytes: ResourceAmount::Known(value),
                ..ResourceVector::zero()
            },
            |value| ResourceVector {
                device_bytes: ResourceAmount::Known(value),
                ..ResourceVector::zero()
            },
            |value| ResourceVector {
                unified_bytes: ResourceAmount::Known(value),
                ..ResourceVector::zero()
            },
        ];

        for vector in vectors {
            let provider = Arc::new(TestProvider {
                snapshot: PhysicalCapacitySnapshot {
                    capacity: vector(100),
                    // Sixty units are already owned outside the authority.
                    available: vector(40),
                    source: CapacitySource::Test,
                },
            });
            let authority = Arc::new(ResourceAuthority::new(provider));
            let first = authority
                .reserve(
                    ReservationOwner::new(ReservationClass::Model, "first"),
                    vector(30),
                )
                .unwrap();

            // The total ledger would allow fifty units, but only forty units
            // of physical headroom exist and thirty are already pending.
            assert!(matches!(
                authority.reserve(
                    ReservationOwner::new(ReservationClass::Model, "second"),
                    vector(20),
                ),
                Err(Error::Overloaded(_))
            ));
            assert_eq!(authority.snapshot().reserved, vector(30));

            drop(first);
            let second = authority
                .reserve(
                    ReservationOwner::new(ReservationClass::Model, "second"),
                    vector(20),
                )
                .unwrap();
            drop(second);
            assert_eq!(authority.snapshot().reserved, vector(0));
        }
    }

    #[test]
    fn mixed_materialized_and_pending_claims_share_live_headroom() {
        let provider = Arc::new(LiveProvider {
            capacity: 100,
            available: AtomicU64::new(40),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let materialized = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "materialized"),
                slots(20),
            )
            .unwrap();
        provider.set_available(30);
        materialized.reconcile_materialized(slots(20)).unwrap();

        let pending = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "pending"),
                slots(15),
            )
            .unwrap();
        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Request, "too-large"),
                slots(16),
            ),
            Err(Error::Overloaded(_))
        ));
        let fitting = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "fitting"),
                slots(15),
            )
            .unwrap();

        assert_eq!(authority.snapshot().reserved, slots(50));
        drop((fitting, pending, materialized));
        assert_eq!(authority.snapshot().reserved, slots(0));
    }

    #[test]
    fn lease_growth_accounts_for_other_pending_reservations() {
        let provider = Arc::new(LiveProvider {
            capacity: 100,
            available: AtomicU64::new(50),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let mut materialized = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "materialized"),
                slots(10),
            )
            .unwrap();
        provider.set_available(40);
        materialized.reconcile_materialized(slots(10)).unwrap();
        let _pending = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "pending"),
                slots(15),
            )
            .unwrap();

        // Growing the materialized lease to 36 adds 26 pending units. Together
        // with the other 15-unit claim that would consume 41 live units.
        assert!(matches!(
            materialized.resize(slots(36)),
            Err(Error::Overloaded(_))
        ));
        materialized.resize(slots(35)).unwrap();
        assert_eq!(authority.snapshot().reserved, slots(50));
    }

    #[test]
    fn materialized_reconciliation_cannot_expand_authorization_after_allocation() {
        let provider = Arc::new(LiveProvider {
            capacity: 10,
            available: AtomicU64::new(10),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let mut cache = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "session"),
                slots(2),
            )
            .unwrap();

        provider.set_available(1);
        assert!(matches!(cache.resize(slots(4)), Err(Error::Overloaded(_))));
        assert_eq!(cache.resources(), slots(2));
        assert_eq!(authority.snapshot().reserved, slots(2));

        // Observation cannot retroactively authorize an allocation that was
        // not reserved before physical growth, even if the provider now sees
        // the allocation and reports less live headroom.
        assert!(matches!(
            cache.reconcile_materialized(slots(4)),
            Err(Error::InferenceError(_))
        ));
        assert_eq!(cache.resources(), slots(2));
        assert_eq!(authority.snapshot().reserved, slots(2));

        cache.reconcile_materialized(slots(2)).unwrap();
        assert!(matches!(
            cache.resize(slots(1)),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(cache.resources(), slots(2));
        assert_eq!(authority.snapshot().reserved, slots(2));
    }

    #[test]
    fn materialized_usage_preserves_authorized_future_growth() {
        let provider = Arc::new(LiveProvider {
            capacity: 10,
            available: AtomicU64::new(10),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let cache = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "bounded-session"),
                slots(8),
            )
            .unwrap();

        provider.set_available(8);
        cache.record_materialized_usage(slots(2)).unwrap();

        assert_eq!(cache.resources(), slots(8));
        assert_eq!(authority.snapshot().reserved, slots(8));
        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Request, "double-spend"),
                slots(3),
            ),
            Err(Error::Overloaded(_))
        ));

        cache.record_materialized_usage(slots(8)).unwrap();
        assert!(matches!(
            cache.record_materialized_usage(slots(9)),
            Err(Error::InferenceError(_))
        ));
        assert_eq!(authority.snapshot().reserved, slots(8));
    }

    #[test]
    fn materialization_diagnostics_identify_domains_and_preserve_ledger() {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(LiveProvider {
            capacity: 10,
            available: AtomicU64::new(10),
        })));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "private caller data"),
                slots(8),
            )
            .unwrap();
        lease.record_materialized_usage(slots(2)).unwrap();
        let observed = ResourceVector {
            host_bytes: ResourceAmount::Known(1),
            compute_slots: ResourceAmount::Known(9),
            ..ResourceVector::zero()
        };
        for (operation, error) in [
            (
                "record_materialized_usage",
                lease.record_materialized_usage(observed).unwrap_err(),
            ),
            (
                "prepare_materialized_release",
                lease.prepare_materialized_release(observed).unwrap_err(),
            ),
        ] {
            let Error::InferenceError(message) = error else {
                panic!("materialization overflow must remain an inference error");
            };
            assert!(message
                .starts_with("materialized resource usage exceeds its authorized reservation:"));
            assert!(message.contains("reservation_id=1"));
            assert!(message.contains("owner_class=Some(Request)"));
            assert!(message.contains(&format!("operation={operation}")));
            assert!(message.contains(&format!("authorized={:?}", slots(8))));
            assert!(message.contains(&format!("observed={observed:?}")));
            assert!(message.contains("exceeded_domains=[\"host_bytes\", \"compute_slots\"]"));
            assert!(!message.contains("private caller data"));
            let state = authority.state.lock().unwrap();
            assert_eq!(state.materialized.get(&ReservationId(1)), Some(&slots(2)));
            assert_eq!(state.ledger.used(), slots(8));
        }
        drop(lease);
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[test]
    fn post_allocation_observation_rejects_materialized_decrease() {
        let provider = Arc::new(LiveProvider {
            capacity: 200,
            available: AtomicU64::new(100),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Pipeline, "retained-growth"),
                slots(100),
            )
            .unwrap();

        provider.set_available(0);
        lease.record_materialized_usage(slots(100)).unwrap();
        provider.set_available(50);

        // The owner's accounting bug, not the client's input.
        assert!(matches!(
            lease.record_materialized_usage(slots(50)),
            Err(Error::InferenceError(_))
        ));
        assert_eq!(authority.snapshot().reserved, slots(100));
    }

    #[test]
    fn ordered_release_allows_smaller_replacement_materialization() {
        let provider = Arc::new(LiveProvider {
            capacity: 200,
            available: AtomicU64::new(100),
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Pipeline, "replace-materialization"),
                slots(100),
            )
            .unwrap();

        lease.record_materialized_usage(slots(100)).unwrap();
        lease.prepare_materialized_release(slots(0)).unwrap();
        lease.record_materialized_usage(slots(50)).unwrap();

        assert_eq!(authority.snapshot().reserved, slots(100));
        assert_eq!(authority.snapshot().reservations, 1);
    }

    #[test]
    fn ordered_materialized_release_cannot_double_spend_external_headroom() {
        use std::sync::Barrier;

        let provider = Arc::new(LiveProvider {
            capacity: 200,
            // The other hundred units are owned outside this authority.
            available: AtomicU64::new(100),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let retained = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Pipeline, "retained-growth"),
                slots(100),
            )
            .unwrap();
        provider.set_available(0);
        retained.record_materialized_usage(slots(100)).unwrap();

        let release_started = Arc::new(Barrier::new(2));
        let racing_authority = authority.clone();
        let racing_barrier = release_started.clone();
        let racing_admission = std::thread::spawn(move || {
            racing_barrier.wait();
            racing_authority.reserve(
                ReservationOwner::new(ReservationClass::Request, "racing-admission"),
                slots(50),
            )
        });

        retained.prepare_materialized_release(slots(50)).unwrap();
        provider.set_available(50);
        release_started.wait();

        assert!(matches!(
            racing_admission.join().unwrap(),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(authority.snapshot().reserved, slots(100));
        assert_eq!(authority.snapshot().reservations, 1);
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex as StdMutex;

    /// Serializes every test that reserves against the process-global
    /// [`ResourceAuthority`] registry: coordinator authorities are shared
    /// across `RuntimeService` instances, load transactions hold their
    /// variant's full residency estimate (multi-GiB for GGUF families), and
    /// the injected-OOM ladder tests temporarily poison the authority. Two
    /// such tests overlapping can reject an unrelated concurrent admission
    /// with `Overloaded` on capacity-constrained runners. Conversely, any
    /// `RuntimeService` unload that leaves its own runtime drained clears
    /// the shared poison, so an unserialized unload test can erase another
    /// test's poison mid-assertion. Tests whose admission, load, or unload
    /// path consults the global authority must hold this lock for their
    /// whole body.
    pub(crate) fn authority_test_lock() -> &'static StdMutex<()> {
        static LOCK: StdMutex<()> = StdMutex::new(());
        &LOCK
    }
}

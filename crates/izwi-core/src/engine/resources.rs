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

    fn positive_growth_over(self, current: Self) -> Result<Self> {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceAuthoritySnapshot {
    pub physical: PhysicalCapacitySnapshot,
    pub reserved: ResourceVector,
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

impl AuthorityState {
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
        ResourceAuthoritySnapshot {
            physical,
            reserved: state.ledger.used(),
            reservations: state.owners.len(),
        }
    }

    /// Permanently fail new work after a backend-fatal asynchronous error.
    /// Metal command-buffer OOM can leave queued command/fence bookkeeping in
    /// an unusable state, so process/device recreation is required.
    pub(crate) fn poison(&self, reason: impl Into<String>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.poisoned.get_or_insert_with(|| reason.into());
    }

    #[cfg(test)]
    pub(crate) fn poison_reason(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .poisoned
            .clone()
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
    /// read the now-zero host alias from that normalized vector.
    pub(crate) fn planning_headroom_bytes(&self, backend: BackendKind) -> Result<ResourceAmount> {
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
                    return Err(Error::Overloaded(format!(
                        "insufficient live physical capacity for {}: new_pending={pending:?}, existing_pending={existing_pending:?}, live_claim={live_claim:?}, physical_available={:?}, physical_capacity={:?}",
                        owner.key, physical.available, physical.capacity
                    )));
                }
                state.ledger.reserve(resources)?
            }
            CapacityEnforcement::Advisory => {
                state.ledger.update_capacity(advisory_ledger_capacity(
                    self.provider.configured_budget(),
                    self.domain_policy,
                ));
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
                return Err(Error::Overloaded(format!(
                    "insufficient live physical capacity for resource lease growth: next_pending={next_pending:?}, other_pending={other_pending:?}, live_claim={live_claim:?}, physical_available={:?}, physical_capacity={:?}",
                    physical.available, physical.capacity
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
            return Err(Error::InferenceError(
                "materialized resource usage exceeds its authorized reservation".to_string(),
            ));
        }
        if !current.fits_within(resources) {
            return Err(Error::InvalidInput(
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
            return Err(Error::InferenceError(
                "materialized resource usage exceeds its authorized reservation".to_string(),
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

    pub(crate) fn prepare_materialized_release(&self, resources: ResourceVector) -> Result<()> {
        self.lease.prepare_materialized_release(resources)
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
        let message = error.to_string();
        assert!(message.contains("new_pending="));
        assert!(message.contains("existing_pending="));
        assert!(message.contains("physical_available="));
        assert!(message.contains("physical_capacity="));
        assert_eq!(authority.snapshot().reserved, slots(0));
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

        assert!(matches!(
            lease.record_materialized_usage(slots(50)),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(authority.snapshot().reserved, slots(100));
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

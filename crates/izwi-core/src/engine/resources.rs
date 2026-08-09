//! Backend-neutral resource estimates, reservations, and reconciliation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    capacity_used: ResourceVector,
    reservations: HashMap<ReservationId, LedgerReservation>,
    next_id: u64,
}

#[derive(Debug, Clone, Copy)]
struct LedgerReservation {
    resources: ResourceVector,
    capacity_charged: bool,
}

impl ResourceLedger {
    pub fn new(capacity: ResourceVector) -> Self {
        Self {
            capacity,
            used: ResourceVector::zero(),
            capacity_used: ResourceVector::zero(),
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
        self.reserve_internal(resources, true)
    }

    fn track(&mut self, resources: ResourceVector) -> Result<ResourceReservation> {
        self.reserve_internal(resources, false)
    }

    fn reserve_internal(
        &mut self,
        resources: ResourceVector,
        capacity_charged: bool,
    ) -> Result<ResourceReservation> {
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "resource reservation contains an unresolved quantity".to_string(),
            ));
        }
        let used = self.used.checked_add(resources)?;
        let capacity_used = if capacity_charged {
            self.capacity_used.checked_add(resources)?
        } else {
            self.capacity_used
        };
        if !capacity_used.fits_within(self.capacity) {
            return Err(Error::Overloaded(
                "requested resources exceed available capacity".to_string(),
            ));
        }
        let id = ReservationId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.reservations.insert(
            id,
            LedgerReservation {
                resources,
                capacity_charged,
            },
        );
        self.used = used;
        self.capacity_used = capacity_used;
        Ok(ResourceReservation { id, resources })
    }

    pub fn release(&mut self, id: ReservationId) -> Result<bool> {
        let Some(reservation) = self.reservations.remove(&id) else {
            return Ok(false);
        };
        self.used = self.used.checked_sub(reservation.resources)?;
        if reservation.capacity_charged {
            self.capacity_used = self.capacity_used.checked_sub(reservation.resources)?;
        }
        Ok(true)
    }

    pub fn resize(&mut self, id: ReservationId, resources: ResourceVector) -> Result<bool> {
        self.resize_internal(id, resources, true)
    }

    fn resize_tracked(&mut self, id: ReservationId, resources: ResourceVector) -> Result<bool> {
        self.resize_internal(id, resources, false)
    }

    fn resize_internal(
        &mut self,
        id: ReservationId,
        resources: ResourceVector,
        capacity_charged: bool,
    ) -> Result<bool> {
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "resource reservation contains an unresolved quantity".to_string(),
            ));
        }
        let Some(current) = self.reservations.get(&id).copied() else {
            return Ok(false);
        };
        if current.capacity_charged != capacity_charged {
            return Err(Error::InferenceError(
                "resource reservation enforcement cannot change during resize".to_string(),
            ));
        }
        let used = self
            .used
            .checked_sub(current.resources)?
            .checked_add(resources)?;
        let capacity_used = if capacity_charged {
            self.capacity_used
                .checked_sub(current.resources)?
                .checked_add(resources)?
        } else {
            self.capacity_used
        };
        if !capacity_used.fits_within(self.capacity) {
            return Err(Error::Overloaded(
                "resized resources exceed available capacity".to_string(),
            ));
        }
        self.reservations.insert(
            id,
            LedgerReservation {
                resources,
                capacity_charged,
            },
        );
        self.used = used;
        self.capacity_used = capacity_used;
        Ok(true)
    }

    fn reservation(&self, id: ReservationId) -> Option<ResourceVector> {
        self.reservations
            .get(&id)
            .map(|reservation| reservation.resources)
    }

    fn capacity_charged(&self, id: ReservationId) -> Option<bool> {
        self.reservations
            .get(&id)
            .map(|reservation| reservation.capacity_charged)
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
}

impl AuthorityState {
    fn pending_resources(&self) -> Result<ResourceVector> {
        self.ledger.reservations.iter().try_fold(
            ResourceVector::zero(),
            |pending, (id, reservation)| {
                let materialized = self
                    .materialized
                    .get(id)
                    .copied()
                    .unwrap_or_else(ResourceVector::zero);
                pending.checked_add(reservation.resources.positive_growth_over(materialized)?)
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationEnforcement {
    Guarded,
    Tracked,
}

/// One transactional authority for every physical-memory consumer on a backend.
#[derive(Debug)]
pub struct ResourceAuthority {
    provider: Arc<dyn PhysicalCapacityProvider>,
    domain_policy: ResourceDomainPolicy,
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
        Self::with_domain_policy(provider, ResourceDomainPolicy::Independent)
    }

    pub(crate) fn new_shared_host_unified(provider: Arc<dyn PhysicalCapacityProvider>) -> Self {
        Self::with_domain_policy(provider, ResourceDomainPolicy::SharedHostUnified)
    }

    fn with_domain_policy(
        provider: Arc<dyn PhysicalCapacityProvider>,
        domain_policy: ResourceDomainPolicy,
    ) -> Self {
        let capacity =
            normalize_physical_resource_domains(provider.snapshot().capacity, domain_policy);
        Self {
            provider,
            domain_policy,
            state: Mutex::new(AuthorityState {
                ledger: ResourceLedger::new(capacity),
                owners: HashMap::new(),
                materialized: HashMap::new(),
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
        state.ledger.update_capacity(physical.capacity);
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
    /// Live headroom is charged only for future, unmaterialized growth.
    pub fn reserve_with_initial_materialized(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
        materialized: ResourceVector,
    ) -> Result<ResourceLease> {
        self.reserve_internal(
            owner,
            resources,
            materialized,
            ReservationEnforcement::Guarded,
        )
    }

    /// Track a CPU or unified-memory model authorization without treating a
    /// volatile host-headroom sample as a hard load gate. The model remains in
    /// the authority ledger, pending-growth accounting, lifecycle slots, and
    /// telemetry; only its own physical/ledger admission is advisory.
    pub(crate) fn track_model(
        self: &Arc<Self>,
        key: impl Into<String>,
        resources: ResourceVector,
    ) -> Result<ResourceLease> {
        self.track_advisory(
            ReservationOwner::new(ReservationClass::Model, key),
            resources,
        )
    }

    /// Track an advisory CPU/unified-memory claim in the resource ledger while
    /// allowing its authorization to follow exact observed pooled storage.
    pub(crate) fn track_advisory(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
    ) -> Result<ResourceLease> {
        self.reserve_internal(
            owner,
            resources,
            ResourceVector::zero(),
            ReservationEnforcement::Tracked,
        )
    }

    fn reserve_internal(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
        materialized: ResourceVector,
        enforcement: ReservationEnforcement,
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
        let reservation = match enforcement {
            ReservationEnforcement::Guarded => {
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
            ReservationEnforcement::Tracked => state.ledger.track(resources)?,
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
        let capacity_charged = state.ledger.capacity_charged(id).ok_or_else(|| {
            Error::InferenceError("resource reservation enforcement is missing".to_string())
        })?;
        let resized = if capacity_charged {
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
            state.ledger.resize_tracked(id, resources)?
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
    fn tracked_model_bypasses_its_own_capacity_gate_but_remains_accounted() {
        let provider = Arc::new(LiveProvider {
            capacity: 2,
            available: AtomicU64::new(0),
        });
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let model = authority.track_model("model", slots(3)).unwrap();

        assert_eq!(authority.snapshot().reserved, slots(3));
        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Request, "during-load"),
                slots(1),
            ),
            Err(Error::Overloaded(_))
        ));

        // Once allocation is materialized, it is visible to the physical
        // provider rather than charged as pending growth. Guarded work remains
        // independently bounded by live and capacity-enforced headroom.
        provider.set_available(1);
        model.reconcile_materialized(slots(3)).unwrap();
        let request = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "after-load"),
                slots(1),
            )
            .unwrap();
        assert_eq!(authority.snapshot().reserved, slots(4));

        drop((request, model));
        assert_eq!(authority.snapshot().reserved, slots(0));
    }

    #[test]
    fn tracked_model_resize_and_release_remain_transactional() {
        let provider = Arc::new(LiveProvider {
            capacity: 1,
            available: AtomicU64::new(0),
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        let mut model = authority.track_model("model", slots(3)).unwrap();

        model.resize(slots(4)).unwrap();
        assert_eq!(authority.snapshot().reserved, slots(4));
        drop(model);
        assert_eq!(authority.snapshot().reserved, slots(0));
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

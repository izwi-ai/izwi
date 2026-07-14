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

    pub fn reserve(&mut self, resources: ResourceVector) -> Result<ResourceReservation> {
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(
                "resource reservation contains an unresolved quantity".to_string(),
            ));
        }
        let candidate = self.used.checked_add(resources)?;
        if !candidate.fits_within(self.capacity) {
            return Err(Error::Overloaded(
                "requested resources exceed available capacity".to_string(),
            ));
        }
        let id = ReservationId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.reservations.insert(id, resources);
        self.used = candidate;
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
        let candidate = self.used.checked_sub(current)?.checked_add(resources)?;
        if !candidate.fits_within(self.capacity) {
            return Err(Error::Overloaded(
                "resized resources exceed available capacity".to_string(),
            ));
        }
        self.reservations.insert(id, resources);
        self.used = candidate;
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
            |pending, (id, reserved)| {
                let materialized = self
                    .materialized
                    .get(id)
                    .copied()
                    .unwrap_or_else(ResourceVector::zero);
                pending.checked_add(reserved.positive_growth_over(materialized)?)
            },
        )
    }
}

/// One transactional authority for every physical-memory consumer on a backend.
#[derive(Debug)]
pub struct ResourceAuthority {
    provider: Arc<dyn PhysicalCapacityProvider>,
    state: Mutex<AuthorityState>,
}

impl ResourceAuthority {
    pub fn new(provider: Arc<dyn PhysicalCapacityProvider>) -> Self {
        let capacity = provider.snapshot().capacity;
        Self {
            provider,
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
        let physical = self.provider.snapshot();
        ResourceAuthoritySnapshot {
            physical,
            reserved: state.ledger.used(),
            reservations: state.owners.len(),
        }
    }

    pub fn reserve(
        self: &Arc<Self>,
        owner: ReservationOwner,
        resources: ResourceVector,
    ) -> Result<ResourceLease> {
        if !resources.is_fully_known() {
            return Err(Error::InvalidInput(format!(
                "resource reservation for {} contains an unresolved quantity",
                owner.key
            )));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        // Serialize the observation with ledger mutation. `available` already
        // excludes materialized allocations, but it cannot see reservations
        // that have not allocated yet. Charge every pending claim against live
        // headroom so concurrent reservations cannot spend it more than once.
        let physical = self.provider.snapshot();
        let live_claim = state.pending_resources()?.checked_add(resources)?;
        if !live_claim.fits_within(physical.available) {
            return Err(Error::Overloaded(format!(
                "insufficient live physical capacity for {}",
                owner.key
            )));
        }
        let reservation = state.ledger.reserve(resources)?;
        state.owners.insert(reservation.id, owner);
        state
            .materialized
            .insert(reservation.id, ResourceVector::zero());
        Ok(ResourceLease {
            authority: self.clone(),
            id: Some(reservation.id),
            resources,
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
        let current_pending = current.positive_growth_over(materialized)?;
        let next_pending = resources.positive_growth_over(materialized)?;
        let other_pending = state.pending_resources()?.checked_sub(current_pending)?;
        let live_claim = other_pending.checked_add(next_pending)?;
        let physical = self.provider.snapshot();
        if !live_claim.fits_within(physical.available) {
            return Err(Error::Overloaded(
                "insufficient live physical capacity for resource lease growth".to_string(),
            ));
        }
        if !state.ledger.resize(id, resources)? {
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
        if !resources.fits_within(reserved) {
            return Err(Error::InferenceError(
                "materialized resource usage exceeds its authorized reservation".to_string(),
            ));
        }
        state.materialized.insert(id, resources);
        Ok(())
    }
}

#[derive(Debug)]
pub struct ResourceLease {
    authority: Arc<ResourceAuthority>,
    id: Option<ReservationId>,
    resources: ResourceVector,
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
    /// physical growth.
    pub fn reconcile_materialized(&self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.record_materialized_usage(id, resources)
    }

    /// Record the portion of this reservation that is physically allocated
    /// without relinquishing any of the capacity authorized for future growth.
    pub fn record_materialized_usage(&self, resources: ResourceVector) -> Result<()> {
        let id = self.id.ok_or_else(|| {
            Error::InferenceError("resource lease is no longer active".to_string())
        })?;
        self.authority.record_materialized_usage(id, resources)
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
    fn live_capacity_failure_is_transactional() {
        let provider = Arc::new(TestProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: slots(2),
                available: slots(0),
                source: CapacitySource::Test,
            },
        });
        let authority = Arc::new(ResourceAuthority::new(provider));
        assert!(authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "request"),
                slots(1),
            )
            .is_err());
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
}

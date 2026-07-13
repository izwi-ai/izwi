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
    pub capacity: ResourceVector,
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
            }),
        }
    }

    pub fn snapshot(&self) -> ResourceAuthoritySnapshot {
        let physical = self.provider.snapshot();
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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
        let physical = self.provider.snapshot();
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("resource authority mutex poisoned".to_string()))?;
        let live_candidate = state.ledger.used().checked_add(resources)?;
        if !live_candidate.fits_within(physical.available) {
            return Err(Error::Overloaded(format!(
                "insufficient live physical capacity for {}",
                owner.key
            )));
        }
        let reservation = state.ledger.reserve(resources)?;
        state.owners.insert(reservation.id, owner);
        Ok(ResourceLease {
            authority: self.clone(),
            id: Some(reservation.id),
            resources,
        })
    }

    fn release(&self, id: ReservationId) {
        if let Ok(mut state) = self.state.lock() {
            state.owners.remove(&id);
            let _ = state.ledger.release(id);
        }
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

    #[derive(Debug)]
    struct TestProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for TestProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
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
}

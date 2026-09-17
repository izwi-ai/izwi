//! Cluster capacity coordination for multi-gateway fleets.
//!
//! When one gateway serves a fleet, the worker registry's local dispatch
//! reservations are sufficient: every admission decision happens in one
//! process. When several gateways share workers, each gateway additionally
//! publishes short-lived capacity claims to the shared fleet store so peer
//! selection steers away from workers that are already spoken for.
//!
//! Authority stays exactly where it was: the worker atomically accepts or
//! rejects every invocation, and a lost fleet race only costs one alternate
//! dispatch through the existing bounded retry path. Claims expire by TTL,
//! so a crashed gateway stops consuming cluster capacity with no recovery
//! protocol at all — expiry *is* the crash recovery.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use izwi_serving_protocol::WorkerStatus;

use crate::{
    batch_runtime::store::BatchRuntimeStore,
    worker_registry::{FleetCapacityView, WorkerInstanceKey},
};

/// Claim TTL: long enough to cover select-to-invoke latency, short enough
/// that a crashed gateway's claims stop shadowing capacity quickly.
pub const FLEET_CLAIM_TTL: Duration = Duration::from_secs(30);

/// Locally cached fresh view of peer gateways' cluster claims, refreshed on
/// the worker status-poller cadence. Selection reads this synchronously and
/// never performs I/O under the registry lock.
#[derive(Debug, Clone, Default)]
pub struct FleetCapacitySnapshot {
    inner: Arc<Mutex<HashMap<(String, String), u64>>>,
}

impl FleetCapacitySnapshot {
    pub fn set_worker_claims(&self, worker_id: &str, incarnation_id: &str, claims: u64) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.insert((worker_id.to_string(), incarnation_id.to_string()), claims);
        }
    }
}

impl FleetCapacityView for FleetCapacitySnapshot {
    fn cluster_claims(&self, worker: &WorkerInstanceKey) -> u64 {
        self.inner
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .get(&(
                        worker.worker_id.as_str().to_string(),
                        worker.incarnation_id.as_str().to_string(),
                    ))
                    .copied()
            })
            .unwrap_or(0)
    }
}

/// Best-effort visibility handle for one dispatch's cluster claim. Dropping
/// the guard releases the claim on a runtime if one is available; otherwise
/// the claim expires by TTL, which is the designed crash-recovery path.
pub struct FleetClaimGuard {
    coordinator: Arc<FleetCoordinator>,
    claim_id: String,
}

impl Drop for FleetClaimGuard {
    fn drop(&mut self) {
        let coordinator = Arc::clone(&self.coordinator);
        let claim_id = self.claim_id.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = coordinator.release(&claim_id).await;
            });
        }
    }
}

#[derive(Debug, Clone)]
pub struct FleetCoordinator {
    store: BatchRuntimeStore,
    gateway_id: String,
    claim_ttl: Duration,
    snapshot: FleetCapacitySnapshot,
}

impl FleetCoordinator {
    pub fn new(store: BatchRuntimeStore, gateway_id: String) -> Self {
        Self {
            store,
            gateway_id,
            claim_ttl: FLEET_CLAIM_TTL,
            snapshot: FleetCapacitySnapshot::default(),
        }
    }

    pub fn gateway_id(&self) -> &str {
        &self.gateway_id
    }

    /// Release every claim still owned by this gateway identity. Called once
    /// at startup: a fresh boot owns nothing, so leftovers from a previous
    /// process under the same operator-set identity are dropped immediately
    /// instead of shadowing cluster capacity until TTL expiry.
    pub async fn release_own_claims(&self) -> u64 {
        self.store
            .release_gateway_claims(&self.gateway_id)
            .await
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn store(&self) -> &BatchRuntimeStore {
        &self.store
    }

    pub fn snapshot_view(&self) -> FleetCapacitySnapshot {
        self.snapshot.clone()
    }

    fn claim_ttl_ms(&self) -> u64 {
        u64::try_from(self.claim_ttl.as_millis()).unwrap_or(30_000)
    }

    /// Claim one unit of cluster capacity for a selected worker. Returns
    /// None when peers already hold every observable credit; the caller
    /// proceeds to invoke anyway and lets the worker arbitrate, so a lost
    /// race degrades to one alternate dispatch rather than a dropped request.
    pub async fn claim(
        self: &Arc<Self>,
        worker: &WorkerInstanceKey,
        available: u32,
    ) -> Option<FleetClaimGuard> {
        let claim_id = self
            .store
            .try_claim_fleet_capacity(
                worker.worker_id.as_str(),
                worker.incarnation_id.as_str(),
                &self.gateway_id,
                available,
                self.claim_ttl_ms(),
            )
            .await
            .ok()??;
        Some(FleetClaimGuard {
            coordinator: Arc::clone(self),
            claim_id,
        })
    }

    async fn release(&self, claim_id: &str) -> bool {
        self.store
            .release_fleet_capacity(claim_id, &self.gateway_id)
            .await
            .unwrap_or(false)
    }

    /// Publish one polled worker status to the shared store and refresh the
    /// local snapshot entry for that worker. Failures retain the previous
    /// view; the worker's direct observations remain authoritative locally.
    pub async fn publish_and_refresh(&self, status: &WorkerStatus) {
        let deployments = serde_json::to_string(&status.deployments).unwrap_or_default();
        let observation = crate::batch_runtime::fleet::FleetWorkerObservation {
            worker_id: status.worker_id.as_str().to_string(),
            node_id: status.node_id.as_str().to_string(),
            incarnation_id: status.incarnation_id.as_str().to_string(),
            status_sequence: status.status_sequence,
            process_state: format!("{:?}", status.process_state),
            available_admission_credits: status.capacity.available_admission_credits,
            active_invocations: status.capacity.active_invocations,
            deployments_json: deployments,
        };
        let _ = self.store.observe_fleet_worker(&observation).await;
        let claims = self
            .store
            .count_live_fleet_claims(status.worker_id.as_str())
            .await
            .unwrap_or_else(|_| {
                self.snapshot.cluster_claims(&WorkerInstanceKey {
                    worker_id: status.worker_id.clone(),
                    incarnation_id: status.incarnation_id.clone(),
                })
            });
        self.snapshot.set_worker_claims(
            status.worker_id.as_str(),
            status.incarnation_id.as_str(),
            claims,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::StoreDatabase;
    use std::sync::atomic::AtomicI64;

    fn coordinator() -> (Arc<FleetCoordinator>, tempfile::TempDir) {
        let root = tempfile::tempdir().unwrap();
        let mut store = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("fleet-coord.sqlite3"),
        ));
        store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
        (
            Arc::new(FleetCoordinator::new(store, "gateway-test".to_string())),
            root,
        )
    }

    fn worker_key() -> WorkerInstanceKey {
        use izwi_serving_protocol::{IncarnationId, WorkerId};
        WorkerInstanceKey {
            worker_id: WorkerId::new("worker-a").unwrap(),
            incarnation_id: IncarnationId::new("inc-a").unwrap(),
        }
    }

    #[tokio::test]
    async fn claim_guard_releases_on_drop_and_snapshot_serves_selection() {
        let (coordinator, _root) = coordinator();
        let key = worker_key();
        assert_eq!(coordinator.snapshot_view().cluster_claims(&key), 0);
        let guard = coordinator.claim(&key, 2).await.expect("claim");
        coordinator
            .snapshot
            .set_worker_claims("worker-a", "inc-a", 1);
        assert_eq!(coordinator.snapshot_view().cluster_claims(&key), 1);
        drop(guard);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            coordinator
                .store
                .count_live_fleet_claims("worker-a")
                .await
                .unwrap(),
            0,
            "dropped guard must release its claim"
        );
    }

    #[tokio::test]
    async fn claim_returns_none_when_peers_hold_all_credits() {
        let (coordinator, _root) = coordinator();
        let key = worker_key();
        let _first = coordinator.claim(&key, 1).await.expect("first claim");
        assert!(
            coordinator.claim(&key, 1).await.is_none(),
            "second claim must lose when one credit is observable"
        );
    }

    #[tokio::test]
    async fn peer_gateway_claims_are_visible_across_coordinators() {
        // Rolling-replacement equivalent: two gateway identities share one
        // coordination store. The replacement sees the predecessor's claims
        // and cannot overspend the same worker credit.
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("shared-fleet.sqlite3");
        let build = |gateway: &str| {
            Arc::new(FleetCoordinator::new(
                BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone())),
                gateway.to_string(),
            ))
        };
        let old = build("gateway-old");
        let new = build("gateway-new");
        let key = worker_key();
        let _held = old.claim(&key, 1).await.expect("predecessor claim");
        assert!(
            new.claim(&key, 1).await.is_none(),
            "replacement gateway must observe the predecessor's live claim"
        );
        assert_eq!(
            new.release_own_claims().await,
            0,
            "a gateway never releases another gateway's claims"
        );
        drop(_held);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            new.claim(&key, 1).await.is_some(),
            "capacity is reusable after the predecessor's claim is released"
        );
    }

    #[tokio::test]
    async fn coordinator_degrades_to_uncoordinated_on_store_outage() {
        // Store outage equivalent: when the coordination database is
        // unreachable, claims fail closed to None and dispatch proceeds
        // uncoordinated (the worker still arbitrates admission).
        let root = tempfile::tempdir().unwrap();
        let dead_path = root.path().join("gone").join("fleet.sqlite3");
        let coordinator = Arc::new(FleetCoordinator::new(
            BatchRuntimeStore::initialize_with_database(StoreDatabase::new(dead_path)),
            "gateway-outage".to_string(),
        ));
        let key = worker_key();
        assert!(
            coordinator.claim(&key, 4).await.is_none(),
            "claims must fail to None, never panic, on store outage"
        );
        assert_eq!(coordinator.release_own_claims().await, 0);
    }
}

//! Cluster capacity coordination for multi-gateway fleets.
//!
//! When one gateway serves a fleet, the worker registry's local dispatch
//! reservations are sufficient: every admission decision happens in one
//! process. When several gateways share workers, each gateway additionally
//! publishes short-lived capacity claims to the shared fleet store so peer
//! selection steers away from workers that are already spoken for.
//!
//! A claim covers only the transit window between selection and the
//! worker's acceptance. Once the worker accepts, its own status reports the
//! work (`available_admission_credits` already excludes it), so the claim is
//! released at acceptance; holding it to the end of the stream would count
//! the same credit twice. A peer samples a worker's status and that worker's
//! claim count in the same poll, so it sees an invocation either as a claim
//! or as consumed credit. The only overlap is the brief gap between the
//! worker answering the status poll and the claim count query; a dispatch
//! that over-steers through it costs one worker-arbitrated alternate. Peer
//! counts exclude this gateway's own claims, which its registry already
//! tracks as local dispatches.
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
    gateway_fleet::{FleetStore, DEFAULT_FLEET_CLAIM_TIMEOUT_MS, DEFAULT_FLEET_CLAIM_TTL_MS},
    worker_registry::{FleetCapacityView, WorkerInstanceKey},
};

/// Bounded batch size for one maintenance-sweep pass, matching the store's
/// bounded reap helper.
pub const FLEET_MAINTENANCE_BATCH: usize = 1024;

/// Cadence of the background maintenance sweep. Slow by design: the reader
/// side already ignores expired rows, so the sweep only reclaims space.
pub const FLEET_MAINTENANCE_INTERVAL: Duration = Duration::from_secs(60);

/// Bound on one background status publish (one upsert plus one count). The
/// publisher runs off the status poller, so this only keeps a hung store
/// from wedging the publisher; the next status then starts a fresh attempt.
const FLEET_PUBLISH_TIMEOUT: Duration = Duration::from_secs(1);

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

/// Latest-value hand-off from one worker's status poller to its background
/// fleet publisher. Publishing never blocks the poller: a newer status
/// replaces one the publisher has not picked up yet, and dropping the handle
/// stops the publisher.
pub struct FleetStatusPublisher {
    latest: tokio::sync::watch::Sender<Option<WorkerStatus>>,
    task: tokio::task::JoinHandle<()>,
}

impl FleetStatusPublisher {
    pub fn publish(&self, status: WorkerStatus) {
        self.latest.send_replace(Some(status));
    }
}

impl Drop for FleetStatusPublisher {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Debug, Clone)]
pub struct FleetCoordinator {
    store: FleetStore,
    gateway_id: String,
    claim_ttl: Duration,
    claim_timeout: Duration,
    snapshot: FleetCapacitySnapshot,
}

impl FleetCoordinator {
    pub fn new(store: FleetStore, gateway_id: String) -> Self {
        Self {
            store,
            gateway_id,
            claim_ttl: Duration::from_millis(DEFAULT_FLEET_CLAIM_TTL_MS),
            claim_timeout: Duration::from_millis(DEFAULT_FLEET_CLAIM_TIMEOUT_MS),
            snapshot: FleetCapacitySnapshot::default(),
        }
    }

    /// Override how long a dispatch waits for its claim
    /// (`IZWI_GATEWAY_FLEET_CLAIM_TIMEOUT_MS`).
    pub fn with_claim_timeout(mut self, claim_timeout: Duration) -> Self {
        self.claim_timeout = claim_timeout;
        self
    }

    /// Override the claim TTL (DS5.3: `IZWI_GATEWAY_FLEET_CLAIM_TTL_MS`).
    /// Unset keeps the historical 30s default.
    pub fn with_claim_ttl(mut self, claim_ttl: Duration) -> Self {
        self.claim_ttl = claim_ttl;
        self
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
    pub fn store(&self) -> &FleetStore {
        &self.store
    }

    /// Opens (and migrates) the store up front, as gateway boot does, so the
    /// first claim is not measured against the lazy first open.
    #[cfg(test)]
    pub async fn opened(self) -> Self {
        self.store.connection().await.expect("fleet store opens");
        self
    }

    pub fn snapshot_view(&self) -> FleetCapacitySnapshot {
        self.snapshot.clone()
    }

    fn claim_ttl_ms(&self) -> u64 {
        u64::try_from(self.claim_ttl.as_millis()).unwrap_or(30_000)
    }

    /// Claim one unit of cluster capacity for a selected worker. Returns
    /// None when peers already hold every observable credit, when the store
    /// fails, or when it does not answer within the claim timeout; the
    /// caller proceeds to invoke anyway and lets the worker arbitrate, so a
    /// lost race or a slow store degrades to worker-authoritative admission
    /// rather than a delayed or dropped request.
    ///
    /// The store call runs on its own task so a claim that lands after the
    /// dispatcher stopped waiting is released at once instead of shadowing a
    /// credit until its TTL. The pool's acquire timeout (and, on PostgreSQL,
    /// the session statement timeout) bounds how long such a task can live.
    pub async fn claim(
        self: &Arc<Self>,
        worker: &WorkerInstanceKey,
        available: u32,
    ) -> Option<FleetClaimGuard> {
        let coordinator = Arc::clone(self);
        let worker_id = worker.worker_id.as_str().to_string();
        let incarnation_id = worker.incarnation_id.as_str().to_string();
        let (claimed, wait) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let claim_id = coordinator
                .store
                .try_claim_fleet_capacity(
                    &worker_id,
                    &incarnation_id,
                    &coordinator.gateway_id,
                    available,
                    coordinator.claim_ttl_ms(),
                )
                .await;
            if let Ok(Some(claim_id)) = claim_id {
                // A dispatcher that timed out dropped the receiver; the
                // returned guard then drops here and releases the claim.
                let _ = claimed.send(FleetClaimGuard {
                    coordinator,
                    claim_id,
                });
            }
        });
        tokio::time::timeout(self.claim_timeout, wait)
            .await
            .ok()?
            .ok()
    }

    /// Start the background refresher of one worker's peer-claim count. The
    /// status poller records each status locally first and then hands it
    /// over here, so a slow or hung coordination store can never delay the
    /// observation that keeps the worker fresh.
    pub fn spawn_status_publisher(self: &Arc<Self>) -> FleetStatusPublisher {
        let (latest, mut pending) = tokio::sync::watch::channel(None::<WorkerStatus>);
        let coordinator = Arc::clone(self);
        let task = tokio::spawn(async move {
            while pending.changed().await.is_ok() {
                let Some(status) = pending.borrow_and_update().clone() else {
                    continue;
                };
                if tokio::time::timeout(
                    FLEET_PUBLISH_TIMEOUT,
                    coordinator.refresh_peer_claims(&status),
                )
                .await
                .is_err()
                {
                    coordinator.forget_peer_claims(&status);
                }
            }
        });
        FleetStatusPublisher { latest, task }
    }

    /// A peer-claim count this gateway cannot refresh is not evidence: fall
    /// back to worker-authoritative admission for that worker.
    fn forget_peer_claims(&self, status: &WorkerStatus) {
        self.snapshot.set_worker_claims(
            status.worker_id.as_str(),
            status.incarnation_id.as_str(),
            0,
        );
    }

    async fn release(&self, claim_id: &str) -> bool {
        self.store
            .release_fleet_capacity(claim_id, &self.gateway_id)
            .await
            .unwrap_or(false)
    }

    /// Refresh the local snapshot of peer gateways' claims on one polled
    /// worker. A failed count clears the worker's peer claims: the worker's
    /// direct observations remain authoritative, and claims only ever steer.
    pub async fn refresh_peer_claims(&self, status: &WorkerStatus) {
        match self
            .store
            .count_live_peer_fleet_claims(status.worker_id.as_str(), &self.gateway_id)
            .await
        {
            Ok(claims) => self.snapshot.set_worker_claims(
                status.worker_id.as_str(),
                status.incarnation_id.as_str(),
                claims,
            ),
            Err(_) => self.forget_peer_claims(status),
        }
    }

    /// One bounded maintenance pass: physically remove expired claims (the
    /// reader-side TTL already stopped counting them). DS5.3 wires this onto
    /// a slow production cadence so a validated fleet profile does not leak
    /// rows; failures are swallowed, consistent with the store-outage
    /// degradation contract. Returns the claims removed.
    pub async fn maintenance_sweep(&self) -> u64 {
        self.store
            .reap_expired_fleet_claims(FLEET_MAINTENANCE_BATCH)
            .await
            .unwrap_or(0)
    }
}

/// Claim timeout for tests that assert a claim lands. The production
/// timeout (100 ms) deliberately gives up on a slow store, and a SQLite
/// write can take that long on a loaded 2 vCPU CI runner, which turned
/// "claim succeeds" assertions into a race with fsync. Tests of the timeout
/// itself keep the production value. 5 s is the configured maximum.
#[cfg(test)]
pub(crate) const TEST_FLEET_CLAIM_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::StoreDatabase;
    use std::sync::atomic::AtomicI64;

    async fn coordinator() -> (Arc<FleetCoordinator>, tempfile::TempDir) {
        let root = tempfile::tempdir().unwrap();
        let mut store =
            FleetStore::new(StoreDatabase::new(root.path().join("fleet-coord.sqlite3")));
        store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
        (
            Arc::new(
                FleetCoordinator::new(store, "gateway-test".to_string())
                    .with_claim_timeout(TEST_FLEET_CLAIM_TIMEOUT)
                    .opened()
                    .await,
            ),
            root,
        )
    }

    /// Build a coordinator with an explicit claim TTL and expose the store's
    /// test clock so tests can advance fleet time through the shared atomic.
    async fn coordinator_with(
        claim_ttl: Duration,
    ) -> (Arc<FleetCoordinator>, Arc<AtomicI64>, tempfile::TempDir) {
        let root = tempfile::tempdir().unwrap();
        let mut store =
            FleetStore::new(StoreDatabase::new(root.path().join("fleet-coord.sqlite3")));
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        (
            Arc::new(
                FleetCoordinator::new(store, "gateway-test".to_string())
                    .with_claim_ttl(claim_ttl)
                    .with_claim_timeout(TEST_FLEET_CLAIM_TIMEOUT)
                    .opened()
                    .await,
            ),
            clock,
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
        let (coordinator, _root) = coordinator().await;
        let key = worker_key();
        assert_eq!(coordinator.snapshot_view().cluster_claims(&key), 0);
        let guard = coordinator.claim(&key, 2).await.expect("claim");
        coordinator
            .snapshot
            .set_worker_claims("worker-a", "inc-a", 1);
        assert_eq!(coordinator.snapshot_view().cluster_claims(&key), 1);
        drop(guard);
        // The drop releases from a spawned task: wait for it to land rather
        // than for a fixed time, which a loaded runner can outlast.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let live = coordinator
                .store
                .count_live_fleet_claims("worker-a")
                .await
                .unwrap();
            if live == 0 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "dropped guard must release its claim ({live} still live)"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn claim_returns_none_when_peers_hold_all_credits() {
        let (coordinator, _root) = coordinator().await;
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
            FleetCoordinator::new(
                FleetStore::new(StoreDatabase::new(db_path.clone())),
                gateway.to_string(),
            )
            .with_claim_timeout(TEST_FLEET_CLAIM_TIMEOUT)
            .opened()
        };
        let old = Arc::new(build("gateway-old").await);
        let new = Arc::new(build("gateway-new").await);
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
    async fn snapshot_counts_peer_claims_but_not_this_gateways_own() {
        // A gateway's in-transit dispatches are already local registry
        // reservations; counting its own claims again would shadow each
        // credit twice until the next poll.
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("shared-fleet.sqlite3");
        let build = |gateway: &str| {
            FleetCoordinator::new(
                FleetStore::new(StoreDatabase::new(db_path.clone())),
                gateway.to_string(),
            )
            .with_claim_timeout(TEST_FLEET_CLAIM_TIMEOUT)
            .opened()
        };
        let own = Arc::new(build("gateway-own").await);
        let peer = Arc::new(build("gateway-peer").await);
        let key = worker_key();
        let _own_claim = own.claim(&key, 4).await.expect("own claim");
        let _peer_claim = peer.claim(&key, 4).await.expect("peer claim");
        let status = WorkerStatus {
            schema_version: izwi_serving_protocol::PROTOCOL_V1,
            worker_id: key.worker_id.clone(),
            node_id: izwi_serving_protocol::NodeId::new("node-a").unwrap(),
            incarnation_id: key.incarnation_id.clone(),
            status_sequence: 1,
            process_state: izwi_serving_protocol::WorkerProcessState::Running,
            deployments: Vec::new(),
            capacity: izwi_serving_protocol::CapacitySnapshot {
                max_active_invocations: 4,
                active_invocations: 0,
                max_queued_invocations: 0,
                queued_invocations: 0,
                max_sessions: 0,
                reserved_sessions: 0,
                available_admission_credits: 4,
                outstanding_cost_units: 0,
            },
        };
        own.refresh_peer_claims(&status).await;
        assert_eq!(own.snapshot_view().cluster_claims(&key), 1);
    }

    #[tokio::test]
    async fn hung_store_claims_degrade_to_none_and_late_claims_release() {
        let root = tempfile::tempdir().unwrap();
        let mut store =
            FleetStore::new(StoreDatabase::new(root.path().join("fleet-coord.sqlite3")));
        let faults = Arc::new(crate::gateway_fleet::FleetFaults::default());
        store.set_faults_for_test(Arc::clone(&faults));
        let coordinator = Arc::new(
            FleetCoordinator::new(store, "gateway-test".to_string())
                .opened()
                .await,
        );
        let key = worker_key();

        faults.hang_fleet_calls();
        let started = std::time::Instant::now();
        assert!(
            coordinator.claim(&key, 4).await.is_none(),
            "a claim the store never answers must degrade to no claim"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the claim must give up at its timeout, waited {:?}",
            started.elapsed()
        );

        // The store answers again: the abandoned claim lands, and since no
        // dispatcher holds it any more it must be released, not left to
        // shadow the credit until its TTL.
        faults.answer_fleet_calls();
        // Let the abandoned call reach the store before sampling it.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut live = u64::MAX;
        for _ in 0..200 {
            live = coordinator
                .store
                .count_live_fleet_claims("worker-a")
                .await
                .unwrap();
            if live == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(live, 0, "a claim that lands after its timeout is released");
        assert!(
            coordinator.claim(&key, 4).await.is_some(),
            "claims resume once the store answers"
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
            FleetStore::new(StoreDatabase::new(dead_path)),
            "gateway-outage".to_string(),
        ));
        let key = worker_key();
        assert!(
            coordinator.claim(&key, 4).await.is_none(),
            "claims must fail to None, never panic, on store outage"
        );
        assert_eq!(coordinator.release_own_claims().await, 0);
    }

    #[tokio::test]
    async fn claim_ttl_override_bounds_the_shadowing_window() {
        let (coordinator, clock, _root) = coordinator_with(Duration::from_millis(500)).await;
        let key = worker_key();
        let _guard = coordinator.claim(&key, 2).await.expect("claim");
        assert_eq!(
            coordinator
                .store()
                .count_live_fleet_claims("worker-a")
                .await
                .unwrap(),
            1,
            "the claim is live while the TTL holds"
        );
        clock.store(1_501, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            coordinator
                .store()
                .count_live_fleet_claims("worker-a")
                .await
                .unwrap(),
            0,
            "the overridden TTL expires the claim"
        );
    }

    #[tokio::test]
    async fn maintenance_sweep_reaps_expired_claims() {
        let (coordinator, clock, _root) = coordinator_with(Duration::from_millis(500)).await;
        let key = worker_key();
        let _guard = coordinator.claim(&key, 2).await.expect("claim");
        // Advance past the claim TTL.
        clock.store(2_000, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            coordinator.maintenance_sweep().await,
            1,
            "the sweep physically removes the expired claim"
        );
        assert_eq!(
            coordinator
                .store()
                .count_live_fleet_claims("worker-a")
                .await
                .unwrap(),
            0
        );
    }
}

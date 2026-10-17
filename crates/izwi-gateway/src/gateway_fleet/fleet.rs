//! Multi-gateway fleet coordination over a shared SQLite or PostgreSQL store.
//!
//! Single-gateway deployments never touch this module. When operators run
//! several gateways against one worker fleet, each gateway publishes its
//! in-transit capacity claims here so peers steer away from workers that
//! are already spoken for. Each gateway polls every worker's status itself;
//! nothing about worker state is shared through the store.
//!
//! Design rules, matching the rest of the runtime store:
//!
//! - Every mutation is one SQL statement or one short transaction, so it is
//!   atomic across gateway processes sharing the database file.
//! - Capacity claims are advisory load signals, never authority. The worker
//!   remains the atomic admission arbiter; a claim that loses the worker
//!   race is simply released. Claims expire by TTL, so a crashed gateway
//!   stops consuming cluster capacity without any explicit recovery.
//! - All inputs are bounded before they reach SQL.

use crate::db::{raw, StoreDatabase};
use anyhow::Context;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, TransactionTrait};
#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, AtomicI64, Ordering},
    Arc,
};

const MAX_FLEET_ID_BYTES: usize = 256;
const MAX_FLEET_BATCH: usize = 512;

fn bounded_maintenance_batch(limit: usize) -> usize {
    limit.clamp(1, MAX_FLEET_BATCH)
}

fn validate_fleet_id(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= MAX_FLEET_ID_BYTES,
        "{name} exceeds {MAX_FLEET_ID_BYTES} bytes"
    );
    Ok(())
}

/// The shared coordination database the gateways of one fleet publish
/// capacity claims to. It holds no job state.
#[derive(Debug, Clone)]
pub struct FleetStore {
    db: StoreDatabase,
    #[cfg(test)]
    test_clock: Option<Arc<AtomicI64>>,
    #[cfg(test)]
    test_faults: Option<Arc<FleetFaults>>,
}

/// Lets a test make fleet calls hang, as on a database host that stopped
/// answering on an established connection.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct FleetFaults {
    hung: AtomicBool,
    answers: tokio::sync::Notify,
}

#[cfg(test)]
impl FleetFaults {
    pub fn hang_fleet_calls(&self) {
        self.hung.store(true, Ordering::SeqCst);
    }

    pub fn answer_fleet_calls(&self) {
        self.hung.store(false, Ordering::SeqCst);
        self.answers.notify_waiters();
    }

    async fn wait_while_hung(&self) {
        loop {
            let answered = self.answers.notified();
            if !self.hung.load(Ordering::SeqCst) {
                return;
            }
            answered.await;
        }
    }
}

impl FleetStore {
    pub fn new(db: StoreDatabase) -> Self {
        Self {
            db: db.for_fleet(),
            #[cfg(test)]
            test_clock: None,
            #[cfg(test)]
            test_faults: None,
        }
    }

    /// Address the coordination store through a SQLite file path, for fleet
    /// tooling and process rigs that inspect a coordination database.
    pub fn from_path(db_path: std::path::PathBuf) -> Self {
        Self::new(StoreDatabase::new(db_path))
    }

    /// Address the coordination store through a bounded database URL
    /// (PostgreSQL fleet profile, DS5).
    pub fn from_url(url: String) -> Self {
        Self::new(StoreDatabase::from_url(url))
    }

    pub async fn connection(&self) -> anyhow::Result<&DatabaseConnection> {
        self.db.connection().await
    }

    #[cfg(test)]
    pub fn set_test_clock(&mut self, clock: Arc<AtomicI64>) {
        self.test_clock = Some(clock);
    }

    #[cfg(test)]
    pub fn set_faults_for_test(&mut self, faults: Arc<FleetFaults>) {
        self.test_faults = Some(faults);
    }

    /// Hangs a fleet coordination call while the test's faults say the
    /// store stopped answering.
    #[cfg(test)]
    async fn fleet_call_fault(&self) {
        if let Some(faults) = &self.test_faults {
            faults.wait_while_hung().await;
        }
    }

    fn fleet_now(&self) -> i64 {
        #[cfg(test)]
        if let Some(clock) = &self.test_clock {
            return clock.load(Ordering::SeqCst);
        }
        self.db.now_millis()
    }

    /// Atomically claim one unit of cluster capacity for a worker when fewer
    /// than `available` live claims exist. Returns the claim ID, or None when
    /// the worker has no observable cluster capacity left.
    ///
    /// SQLite executes one statement whose count-and-insert runs under the
    /// database's single-writer lock, so concurrent gateways cannot overspend
    /// the same credit. PostgreSQL has no such writer serialization: two
    /// concurrent `INSERT ... SELECT` snapshots can both observe the same
    /// claim count and overspend, so the claim runs in a transaction guarded
    /// by a worker-keyed advisory lock (auto-released at commit).
    pub async fn try_claim_fleet_capacity(
        &self,
        worker_id: &str,
        incarnation_id: &str,
        gateway_id: &str,
        available: u32,
        ttl_ms: u64,
    ) -> anyhow::Result<Option<String>> {
        #[cfg(test)]
        self.fleet_call_fault().await;
        validate_fleet_id("worker ID", worker_id)?;
        validate_fleet_id("incarnation ID", incarnation_id)?;
        validate_fleet_id("gateway ID", gateway_id)?;
        if available == 0 {
            return Ok(None);
        }
        let now = self.fleet_now();
        let claim_id = crate::ids::new_uuid();
        let db = self.connection().await?;
        let backend = db.get_database_backend();
        let claim_values = vec![
            claim_id.clone().into(),
            worker_id.to_string().into(),
            incarnation_id.to_string().into(),
            gateway_id.to_string().into(),
            now.into(),
            now.saturating_add(i64::try_from(ttl_ms)?).into(),
            i64::from(available).into(),
        ];
        let affected = if backend == DbBackend::Postgres {
            let tx = db
                .begin()
                .await
                .context("Failed to open fleet claim transaction")?;
            tx.execute_raw(raw::statement(
                &tx,
                "SELECT pg_advisory_xact_lock(hashtextextended(?1, 0))",
                vec![worker_id.to_string().into()],
            )?)
            .await
            .context("Failed to lock fleet worker for claiming")?;
            let inserted = tx
                .execute_raw(raw::statement(
                    &tx,
                    fleet_claim_sql(backend)?,
                    claim_values,
                )?)
                .await
                .context("Failed to claim fleet capacity")?;
            tx.commit()
                .await
                .context("Failed to commit fleet capacity claim")?;
            inserted.rows_affected()
        } else {
            let inserted = db
                .execute_raw(raw::statement(db, fleet_claim_sql(backend)?, claim_values)?)
                .await
                .context("Failed to claim fleet capacity")?;
            inserted.rows_affected()
        };
        Ok(if affected == 1 { Some(claim_id) } else { None })
    }

    /// Release one capacity claim. Only the owning gateway may release it.
    pub async fn release_fleet_capacity(
        &self,
        claim_id: &str,
        gateway_id: &str,
    ) -> anyhow::Result<bool> {
        validate_fleet_id("gateway ID", gateway_id)?;
        let db = self.connection().await?;
        let deleted = db
            .execute_raw(raw::statement(
                db,
                "DELETE FROM fleet_capacity_claims WHERE claim_id = ?1 AND gateway_id = ?2",
                vec![claim_id.to_string().into(), gateway_id.to_string().into()],
            )?)
            .await
            .context("Failed to release fleet capacity claim")?;
        Ok(deleted.rows_affected() == 1)
    }

    /// Release every claim owned by `gateway_id`, returning how many were
    /// released. Called once at gateway startup: a fresh boot owns nothing,
    /// so leftovers from a previous process with the same operator-set
    /// identity are released immediately instead of waiting out their TTL.
    /// The operator must ensure the previous process is actually dead (the
    /// documented drain-before-restart procedure); claims are advisory and
    /// the worker remains the admission arbiter, so a mistaken release only
    /// costs selection precision, never correctness.
    pub async fn release_gateway_claims(&self, gateway_id: &str) -> anyhow::Result<u64> {
        validate_fleet_id("gateway ID", gateway_id)?;
        let db = self.connection().await?;
        let deleted = db
            .execute_raw(raw::statement(
                db,
                r#"
                DELETE FROM fleet_capacity_claims
                WHERE claim_id IN (
                    SELECT claim_id FROM fleet_capacity_claims
                    WHERE gateway_id = ?1
                    ORDER BY created_at ASC
                    LIMIT 1024
                )
                "#,
                vec![gateway_id.to_string().into()],
            )?)
            .await
            .context("Failed to release gateway fleet claims")?;
        Ok(deleted.rows_affected())
    }

    /// Count live (unexpired) claims for one worker.
    pub async fn count_live_fleet_claims(&self, worker_id: &str) -> anyhow::Result<u64> {
        validate_fleet_id("worker ID", worker_id)?;
        let now = self.fleet_now();
        let db = self.connection().await?;
        let count = db
            .query_one_raw(raw::statement(
                db,
                "SELECT COUNT(*) FROM fleet_capacity_claims WHERE worker_id = ?1 AND expires_at > ?2",
                vec![worker_id.to_string().into(), now.into()],
            )?)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Fleet capacity count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        Ok(u64::try_from(count)?)
    }

    /// Count live (unexpired) claims for one worker held by gateways other
    /// than `gateway_id`. A gateway already accounts for its own in-transit
    /// dispatches in its local registry, so counting its own claims again
    /// would shadow the same credit twice.
    pub async fn count_live_peer_fleet_claims(
        &self,
        worker_id: &str,
        gateway_id: &str,
    ) -> anyhow::Result<u64> {
        #[cfg(test)]
        self.fleet_call_fault().await;
        validate_fleet_id("worker ID", worker_id)?;
        validate_fleet_id("gateway ID", gateway_id)?;
        let now = self.fleet_now();
        let db = self.connection().await?;
        let count = db
            .query_one_raw(raw::statement(
                db,
                "SELECT COUNT(*) FROM fleet_capacity_claims WHERE worker_id = ?1 AND gateway_id <> ?2 AND expires_at > ?3",
                vec![
                    worker_id.to_string().into(),
                    gateway_id.to_string().into(),
                    now.into(),
                ],
            )?)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Fleet capacity count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        Ok(u64::try_from(count)?)
    }

    /// Delete expired claims, bounded to `limit` rows. Crash recovery for
    /// gateways that disappear without releasing their claims.
    pub async fn reap_expired_fleet_claims(&self, limit: usize) -> anyhow::Result<u64> {
        let now = self.fleet_now();
        let db = self.connection().await?;
        let deleted = db
            .execute_raw(raw::statement(
                db,
                fleet_reap_sql(db.get_database_backend())?,
                vec![
                    now.into(),
                    i64::try_from(bounded_maintenance_batch(limit))?.into(),
                ],
            )?)
            .await
            .context("Failed to reap expired fleet capacity claims")?;
        Ok(deleted.rows_affected())
    }
}

/// Backend-conditional SQL for fleet coordination. SQLite (a shared file
/// with WAL) and PostgreSQL share these forms; the PostgreSQL lane
/// (`fleet_postgres_execution`) executes them for real.
fn fleet_claim_sql(backend: DbBackend) -> anyhow::Result<&'static str> {
    match backend {
        DbBackend::Sqlite | DbBackend::Postgres => Ok(r#"
            INSERT INTO fleet_capacity_claims (
                claim_id, worker_id, incarnation_id, gateway_id,
                created_at, expires_at
            )
            SELECT ?1, ?2, ?3, ?4, ?5, ?6
            WHERE (
                SELECT COUNT(*) FROM fleet_capacity_claims
                WHERE worker_id = ?2 AND expires_at > ?5
            ) < ?7
        "#),
        backend => anyhow::bail!("Unsupported fleet coordination database backend: {backend:?}"),
    }
}

fn fleet_reap_sql(backend: DbBackend) -> anyhow::Result<&'static str> {
    match backend {
        DbBackend::Sqlite | DbBackend::Postgres => Ok(r#"
            DELETE FROM fleet_capacity_claims
            WHERE claim_id IN (
                SELECT claim_id FROM fleet_capacity_claims
                WHERE expires_at <= ?1
                ORDER BY expires_at ASC
                LIMIT ?2
            )
        "#),
        backend => anyhow::bail!("Unsupported fleet coordination database backend: {backend:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store_at(clock_ms: i64) -> (FleetStore, tempfile::TempDir) {
        let root = tempfile::tempdir().expect("temp dir");
        let mut store = FleetStore::from_path(root.path().join("fleet.sqlite3"));
        store.set_test_clock(Arc::new(AtomicI64::new(clock_ms)));
        (store, root)
    }

    #[tokio::test]
    async fn fleet_capacity_claims_are_atomic_under_concurrency() {
        let (store, _root) = test_store_at(1_000);
        let attempts = 8u32;
        let available = 3u32;
        let mut handles = Vec::new();
        for index in 0..attempts {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .try_claim_fleet_capacity(
                        "worker-a",
                        "inc-1",
                        &format!("gateway-{index}"),
                        available,
                        60_000,
                    )
                    .await
            }));
        }
        let mut acquired = 0u32;
        for handle in handles {
            if handle.await.unwrap().unwrap().is_some() {
                acquired += 1;
            }
        }
        assert_eq!(
            acquired, available,
            "exactly the observable credits may be claimed cluster-wide"
        );
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 3);
        assert!(
            store
                .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-9", available, 60_000)
                .await
                .unwrap()
                .is_none(),
            "no credit remains after the available units are claimed"
        );
    }

    #[tokio::test]
    async fn fleet_claim_release_is_owner_only() {
        let (store, _root) = test_store_at(1_000);
        let claim = store
            .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-a", 2, 60_000)
            .await
            .unwrap()
            .expect("first claim");
        assert!(
            !store
                .release_fleet_capacity(&claim, "gateway-b")
                .await
                .unwrap(),
            "a peer gateway must not release another gateway's claim"
        );
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 1);
        assert!(store
            .release_fleet_capacity(&claim, "gateway-a")
            .await
            .unwrap());
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn peer_claim_count_excludes_the_observing_gateway() {
        let (store, _root) = test_store_at(1_000);
        for gateway in ["gateway-a", "gateway-a", "gateway-b"] {
            store
                .try_claim_fleet_capacity("worker-a", "inc-1", gateway, 4, 60_000)
                .await
                .unwrap()
                .expect("claim");
        }
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 3);
        assert_eq!(
            store
                .count_live_peer_fleet_claims("worker-a", "gateway-a")
                .await
                .unwrap(),
            1,
            "a gateway's own claims are already its local dispatches"
        );
        assert_eq!(
            store
                .count_live_peer_fleet_claims("worker-a", "gateway-b")
                .await
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn fleet_expired_claims_reap_without_explicit_release() {
        let (mut store, _root) = test_store_at(1_000);
        store
            .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-a", 4, 100)
            .await
            .unwrap()
            .expect("claim");
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 1);
        store.set_test_clock(Arc::new(AtomicI64::new(1_101)));
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 0);
        assert_eq!(store.reap_expired_fleet_claims(64).await.unwrap(), 1);
        assert!(
            store
                .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-b", 4, 60_000)
                .await
                .unwrap()
                .is_some(),
            "reaped capacity is reusable"
        );
    }

    #[tokio::test]
    async fn fleet_gateway_restart_releases_only_its_own_claims() {
        let (store, _root) = test_store_at(1_000);
        for _ in 0..2 {
            store
                .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-a", 4, 60_000)
                .await
                .unwrap()
                .expect("gateway-a claim");
        }
        store
            .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-b", 4, 60_000)
            .await
            .unwrap()
            .expect("gateway-b claim");
        assert_eq!(
            store.release_gateway_claims("gateway-a").await.unwrap(),
            2,
            "a restarting gateway releases exactly its own leftovers"
        );
        assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 1);
        assert_eq!(
            store.release_gateway_claims("gateway-a").await.unwrap(),
            0,
            "releasing twice is idempotent"
        );
    }

    #[tokio::test]
    async fn fleet_capacity_claims_are_atomic_across_connections() {
        // Multi-gateway equivalent on one host: two independent store
        // handles (separate pools, as two gateway processes would hold)
        // race claims against the same SQLite file. WAL serializes the
        // writers and the single-statement check-and-insert keeps the
        // cluster-wide total exact.
        let root = tempfile::tempdir().expect("temp dir");
        let db_path = root.path().join("shared-fleet.sqlite3");
        let mut stores = Vec::new();
        for _ in 0..2 {
            let mut store = FleetStore::from_path(db_path.clone());
            store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
            stores.push(store);
        }
        let mut handles = Vec::new();
        // Warm both connections (running migrations) before the race so the
        // test measures claim atomicity, not first-open migration contention.
        // Production gateways serialize the same way at boot.
        for store in &stores {
            store.count_live_fleet_claims("worker-a").await.unwrap();
        }
        for index in 0..8u32 {
            let store = stores[(index % 2) as usize].clone();
            handles.push(tokio::spawn(async move {
                store
                    .try_claim_fleet_capacity(
                        "worker-a",
                        "inc-1",
                        &format!("gateway-{index}"),
                        3,
                        60_000,
                    )
                    .await
            }));
        }
        let mut acquired = 0u32;
        for handle in handles {
            if handle.await.unwrap().unwrap().is_some() {
                acquired += 1;
            }
        }
        assert_eq!(
            acquired, 3,
            "exactly the observable credits may be claimed across connections"
        );
        assert_eq!(
            stores[1].count_live_fleet_claims("worker-a").await.unwrap(),
            3,
            "either connection observes the same cluster total"
        );
    }

    #[tokio::test]
    async fn fleet_inputs_are_bounded() {
        let (store, _root) = test_store_at(1_000);
        assert!(store
            .try_claim_fleet_capacity(&"w".repeat(257), "inc-1", "gateway-a", 1, 60_000)
            .await
            .is_err());
        assert!(store
            .try_claim_fleet_capacity("worker-a", "inc-1", &"g".repeat(257), 1, 60_000)
            .await
            .is_err());
        assert!(store
            .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-a", 0, 60_000)
            .await
            .unwrap()
            .is_none());
    }
}

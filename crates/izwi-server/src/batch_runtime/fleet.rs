//! Multi-gateway fleet coordination over the shared SQLite store.
//!
//! Single-gateway deployments never touch this module. When operators run
//! several gateways against one worker fleet, each gateway publishes its
//! worker observations and capacity claims here so peers get a shared view
//! of cluster state with bounded staleness.
//!
//! Design rules, matching the rest of the runtime store:
//!
//! - Every mutation is one SQL statement or one short transaction, so it is
//!   atomic across gateway processes sharing the database file.
//! - Observation acceptance is monotonic: a higher status sequence wins for
//!   one incarnation, and any incarnation change wins outright. Replacement
//!   incarnations restart their sequence counters, so sequence alone cannot
//!   fence them. Stale resurrections are bounded by the reader-side TTL.
//! - Capacity claims are advisory load signals, never authority. The worker
//!   remains the atomic admission arbiter; a claim that loses the worker
//!   race is simply released. Claims expire by TTL, so a crashed gateway
//!   stops consuming cluster capacity without any explicit recovery.
//! - All inputs are bounded before they reach SQL.

use super::store::BatchRuntimeStore;
use crate::db::raw;
use anyhow::Context;
use sea_orm::ConnectionTrait;

const MAX_FLEET_ID_BYTES: usize = 256;
const MAX_FLEET_DEPLOYMENTS_JSON_BYTES: usize = 64 * 1024;
const MAX_FLEET_BATCH: usize = 512;

fn bounded_maintenance_batch(limit: usize) -> usize {
    limit.min(MAX_FLEET_BATCH).max(1)
}

fn validate_fleet_id(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= MAX_FLEET_ID_BYTES,
        "{name} exceeds {MAX_FLEET_ID_BYTES} bytes"
    );
    Ok(())
}

/// One worker-status publication from a gateway's poll loop.
pub struct FleetWorkerObservation {
    pub worker_id: String,
    pub node_id: String,
    pub incarnation_id: String,
    pub status_sequence: u64,
    pub process_state: String,
    pub available_admission_credits: u32,
    pub active_invocations: u32,
    pub deployments_json: String,
}

/// A fresh cluster-wide view of one worker.
pub struct FleetWorkerView {
    pub worker_id: String,
    pub node_id: String,
    pub incarnation_id: String,
    pub status_sequence: u64,
    pub process_state: String,
    pub available_admission_credits: u32,
    pub active_invocations: u32,
    pub deployments_json: String,
    pub observed_at: i64,
    pub observer_gateway_id: String,
}

impl BatchRuntimeStore {
    fn fleet_now(&self) -> i64 {
        self.now_millis()
    }

    /// Publish one worker observation. Returns true when the row was created
    /// or advanced; false when the stored view is already newer and the
    /// publication was ignored.
    pub async fn observe_fleet_worker(
        &self,
        input: &FleetWorkerObservation,
    ) -> anyhow::Result<bool> {
        validate_fleet_id("worker ID", &input.worker_id)?;
        validate_fleet_id("node ID", &input.node_id)?;
        validate_fleet_id("incarnation ID", &input.incarnation_id)?;
        anyhow::ensure!(
            input.deployments_json.len() <= MAX_FLEET_DEPLOYMENTS_JSON_BYTES,
            "fleet deployments document exceeds bounds"
        );
        let now = self.fleet_now();
        let db = self.connection().await?;
        let updated = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE fleet_worker_observations
                SET node_id = ?1, incarnation_id = ?2, status_sequence = ?3,
                    process_state = ?4, available_admission_credits = ?5,
                    active_invocations = ?6, deployments_json = ?7,
                    observed_at = ?8, observer_gateway_id = ?9
                WHERE worker_id = ?10
                  AND (
                    incarnation_id != ?2
                    OR status_sequence < ?3
                    OR (status_sequence = ?3 AND observed_at <= ?8)
                  )
                "#,
                vec![
                    input.node_id.clone().into(),
                    input.incarnation_id.clone().into(),
                    i64::try_from(input.status_sequence)?.into(),
                    input.process_state.clone().into(),
                    i64::from(input.available_admission_credits).into(),
                    i64::from(input.active_invocations).into(),
                    input.deployments_json.clone().into(),
                    now.into(),
                    fleet_observer_gateway_id().into(),
                    input.worker_id.clone().into(),
                ],
            )?)
            .await
            .context("Failed to publish fleet worker observation")?;
        if updated.rows_affected() == 1 {
            return Ok(true);
        }
        let inserted = db
            .execute_raw(raw::statement(
                db,
                r#"
                INSERT INTO fleet_worker_observations (
                    worker_id, node_id, incarnation_id, status_sequence,
                    process_state, available_admission_credits,
                    active_invocations, deployments_json, observed_at,
                    observer_gateway_id
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT(worker_id) DO NOTHING
                "#,
                vec![
                    input.worker_id.clone().into(),
                    input.node_id.clone().into(),
                    input.incarnation_id.clone().into(),
                    i64::try_from(input.status_sequence)?.into(),
                    input.process_state.clone().into(),
                    i64::from(input.available_admission_credits).into(),
                    i64::from(input.active_invocations).into(),
                    input.deployments_json.clone().into(),
                    now.into(),
                    fleet_observer_gateway_id().into(),
                ],
            )?)
            .await
            .context("Failed to insert fleet worker observation")?;
        Ok(inserted.rows_affected() == 1)
    }

    /// Read the cluster-wide fresh view: rows observed within `ttl_ms`.
    pub async fn read_fresh_fleet_workers(
        &self,
        ttl_ms: u64,
    ) -> anyhow::Result<Vec<FleetWorkerView>> {
        let now = self.fleet_now();
        let db = self.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT worker_id, node_id, incarnation_id, status_sequence,
                       process_state, available_admission_credits,
                       active_invocations, deployments_json, observed_at,
                       observer_gateway_id
                FROM fleet_worker_observations
                WHERE observed_at > ?1
                ORDER BY worker_id ASC
                LIMIT ?2
                "#,
                vec![
                    now.saturating_sub(i64::try_from(ttl_ms)?).into(),
                    i64::try_from(MAX_FLEET_BATCH)?.into(),
                ],
            )?)
            .await
            .context("Failed to read fleet worker observations")?;
        let mut views = Vec::with_capacity(rows.len());
        for row in rows {
            views.push(FleetWorkerView {
                worker_id: row.try_get_by_index(0)?,
                node_id: row.try_get_by_index(1)?,
                incarnation_id: row.try_get_by_index(2)?,
                status_sequence: u64::try_from(row.try_get_by_index::<i64>(3)?)?,
                process_state: row.try_get_by_index(4)?,
                available_admission_credits: u32::try_from(row.try_get_by_index::<i64>(5)?)?,
                active_invocations: u32::try_from(row.try_get_by_index::<i64>(6)?)?,
                deployments_json: row.try_get_by_index(7)?,
                observed_at: row.try_get_by_index(8)?,
                observer_gateway_id: row.try_get_by_index(9)?,
            });
        }
        Ok(views)
    }

    /// Delete observations older than `max_age_ms`, bounded to `limit` rows.
    pub async fn prune_stale_fleet_observations(
        &self,
        max_age_ms: u64,
        limit: usize,
    ) -> anyhow::Result<u64> {
        let now = self.fleet_now();
        let db = self.connection().await?;
        let deleted = db
            .execute_raw(raw::statement(
                db,
                r#"
                DELETE FROM fleet_worker_observations
                WHERE worker_id IN (
                    SELECT worker_id FROM fleet_worker_observations
                    WHERE observed_at <= ?1
                    ORDER BY observed_at ASC
                    LIMIT ?2
                )
                "#,
                vec![
                    now.saturating_sub(i64::try_from(max_age_ms)?).into(),
                    i64::try_from(bounded_maintenance_batch(limit))?.into(),
                ],
            )?)
            .await
            .context("Failed to prune fleet worker observations")?;
        Ok(u64::try_from(deleted.rows_affected())?)
    }

    /// Atomically claim one unit of cluster capacity for a worker when fewer
    /// than `available` live claims exist. Returns the claim ID, or None when
    /// the worker has no observable cluster capacity left. One statement, so
    /// concurrent gateways cannot overspend the same credit.
    pub async fn try_claim_fleet_capacity(
        &self,
        worker_id: &str,
        incarnation_id: &str,
        gateway_id: &str,
        available: u32,
        ttl_ms: u64,
    ) -> anyhow::Result<Option<String>> {
        validate_fleet_id("worker ID", worker_id)?;
        validate_fleet_id("incarnation ID", incarnation_id)?;
        validate_fleet_id("gateway ID", gateway_id)?;
        if available == 0 {
            return Ok(None);
        }
        let now = self.fleet_now();
        let claim_id = crate::ids::new_uuid();
        let db = self.connection().await?;
        let inserted = db
            .execute_raw(raw::statement(
                db,
                r#"
                INSERT INTO fleet_capacity_claims (
                    claim_id, worker_id, incarnation_id, gateway_id,
                    created_at, expires_at
                )
                SELECT ?1, ?2, ?3, ?4, ?5, ?6
                WHERE (
                    SELECT COUNT(*) FROM fleet_capacity_claims
                    WHERE worker_id = ?2 AND expires_at > ?5
                ) < ?7
                "#,
                vec![
                    claim_id.clone().into(),
                    worker_id.to_string().into(),
                    incarnation_id.to_string().into(),
                    gateway_id.to_string().into(),
                    now.into(),
                    now.saturating_add(i64::try_from(ttl_ms)?).into(),
                    i64::from(available).into(),
                ],
            )?)
            .await
            .context("Failed to claim fleet capacity")?;
        Ok(if inserted.rows_affected() == 1 {
            Some(claim_id)
        } else {
            None
        })
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

    /// Delete expired claims, bounded to `limit` rows. Crash recovery for
    /// gateways that disappear without releasing their claims.
    pub async fn reap_expired_fleet_claims(&self, limit: usize) -> anyhow::Result<u64> {
        let now = self.fleet_now();
        let db = self.connection().await?;
        let deleted = db
            .execute_raw(raw::statement(
                db,
                r#"
                DELETE FROM fleet_capacity_claims
                WHERE claim_id IN (
                    SELECT claim_id FROM fleet_capacity_claims
                    WHERE expires_at <= ?1
                    ORDER BY expires_at ASC
                    LIMIT ?2
                )
                "#,
                vec![
                    now.into(),
                    i64::try_from(bounded_maintenance_batch(limit))?.into(),
                ],
            )?)
            .await
            .context("Failed to reap expired fleet capacity claims")?;
        Ok(u64::try_from(deleted.rows_affected())?)
    }
}

fn fleet_observer_gateway_id() -> String {
    std::env::var("IZWI_GATEWAY_ID").unwrap_or_else(|_| "gateway".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::StoreDatabase;
    use std::sync::{atomic::AtomicI64, Arc};

    fn test_store_at(clock_ms: i64) -> (BatchRuntimeStore, tempfile::TempDir) {
        let root = tempfile::tempdir().expect("temp dir");
        let mut store = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("fleet.sqlite3"),
        ));
        store.set_test_clock(Arc::new(AtomicI64::new(clock_ms)));
        (store, root)
    }

    fn observation(sequence: u64, incarnation: &str) -> FleetWorkerObservation {
        FleetWorkerObservation {
            worker_id: "worker-a".to_string(),
            node_id: "node-1".to_string(),
            incarnation_id: incarnation.to_string(),
            status_sequence: sequence,
            process_state: "running".to_string(),
            available_admission_credits: 2,
            active_invocations: 0,
            deployments_json: "{}".to_string(),
        }
    }

    #[tokio::test]
    async fn fleet_observation_is_monotonic_within_one_incarnation() {
        let (store, _root) = test_store_at(1_000);
        assert!(store
            .observe_fleet_worker(&observation(1, "inc-1"))
            .await
            .unwrap());
        assert!(store
            .observe_fleet_worker(&observation(2, "inc-1"))
            .await
            .unwrap());
        assert!(
            !store
                .observe_fleet_worker(&observation(1, "inc-1"))
                .await
                .unwrap(),
            "stale sequence must not overwrite a newer view"
        );
        assert!(
            store.observe_fleet_worker(&observation(2, "inc-1")).await.unwrap(),
            "same-sequence re-publication refreshes the observation timestamp without regressing the view"
        );
        let views = store.read_fresh_fleet_workers(60_000).await.unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].status_sequence, 2);
        assert_eq!(views[0].incarnation_id, "inc-1");
    }

    #[tokio::test]
    async fn fleet_observation_incarnation_change_always_wins() {
        let (store, _root) = test_store_at(1_000);
        assert!(store
            .observe_fleet_worker(&observation(90, "inc-1"))
            .await
            .unwrap());
        assert!(
            store
                .observe_fleet_worker(&observation(1, "inc-2"))
                .await
                .unwrap(),
            "replacement incarnations restart their sequence counters"
        );
        let views = store.read_fresh_fleet_workers(60_000).await.unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].incarnation_id, "inc-2");
        assert_eq!(views[0].status_sequence, 1);
    }

    #[tokio::test]
    async fn fleet_fresh_read_applies_reader_side_ttl() {
        let (mut store, _root) = test_store_at(1_000);
        assert!(store
            .observe_fleet_worker(&observation(1, "inc-1"))
            .await
            .unwrap());
        assert_eq!(
            store.read_fresh_fleet_workers(60_000).await.unwrap().len(),
            1
        );
        store.set_test_clock(Arc::new(AtomicI64::new(61_001)));
        assert_eq!(
            store.read_fresh_fleet_workers(60_000).await.unwrap().len(),
            0,
            "observations older than the TTL are not part of the fresh view"
        );
        assert_eq!(
            store
                .prune_stale_fleet_observations(60_000, 64)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .read_fresh_fleet_workers(u64::try_from(i64::MAX).unwrap())
                .await
                .unwrap()
                .len(),
            0
        );
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
    async fn fleet_inputs_are_bounded() {
        let (store, _root) = test_store_at(1_000);
        let mut oversized = observation(1, "inc-1");
        oversized.worker_id = "w".repeat(257);
        assert!(store.observe_fleet_worker(&oversized).await.is_err());
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

//! DS5.1 execution evidence: the fleet coordination store and the durable
//! store run against real PostgreSQL. The SQLite suites in `fleet.rs` and
//! `db/sqlite.rs` prove the operation semantics; this module proves the
//! dialect-conditional SQL and the promoted column types behave identically
//! on a server backend.
//!
//! Tests are skipped unless the `db-postgres` cargo feature is enabled and
//! `IZWI_TEST_FLEET_PG_URL` points at a *disposable* PostgreSQL database
//! (for example `postgres://localhost/izwi_ds5_test`). Every test drops all
//! tables in the target schema and re-runs the migrations, so the database
//! must not hold state anyone cares about.

use crate::batch_runtime::fleet::FleetWorkerObservation;
use crate::batch_runtime::store::BatchRuntimeStore;
use crate::db::migrator::Migrator;
use crate::db::raw;
use crate::db::StoreDatabase;
use sea_orm::{ConnectionTrait, DatabaseConnection, Statement};
use std::sync::atomic::AtomicI64;
use std::sync::Arc;

const PG_URL_ENV: &str = "IZWI_TEST_FLEET_PG_URL";

/// Every test shares one disposable PostgreSQL database, so the drop +
/// re-migrate setup must not interleave between parallel test threads.
static PG_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn pg_url() -> Option<String> {
    match std::env::var(PG_URL_ENV) {
        Ok(url) => Some(url),
        Err(_) => {
            eprintln!("skipping: {PG_URL_ENV} is not set");
            None
        }
    }
}

type PgTestGuard = tokio::sync::MutexGuard<'static, ()>;

async fn fresh_store() -> Option<(BatchRuntimeStore, PgTestGuard)> {
    let guard = PG_TEST_LOCK.lock().await;
    let url = pg_url()?;
    let store = BatchRuntimeStore::initialize_with_database(StoreDatabase::from_url(url));
    let connection = store.connection().await.expect("postgres connection opens");
    drop_all_tables(connection).await.expect("tables drop");
    Migrator::up(connection).await.expect("migrations run");
    Some((store, guard))
}

async fn drop_all_tables(db: &DatabaseConnection) -> anyhow::Result<()> {
    let rows = db
        .query_all_raw(Statement::from_string(
            db.get_database_backend(),
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()".to_string(),
        ))
        .await?;
    for row in rows {
        let table: String = row.try_get_by_index(0)?;
        db.execute_unprepared(&format!("DROP TABLE IF EXISTS \"{table}\" CASCADE"))
            .await?;
    }
    Ok(())
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

async fn column_type(db: &DatabaseConnection, table: &str, column: &str) -> String {
    let row = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT data_type FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ?1 AND column_name = ?2",
                vec![table.into(), column.into()],
            )
            .expect("column statement builds"),
        )
        .await
        .expect("column query runs")
        .expect("column exists");
    row.try_get_by_index(0).expect("data_type column")
}

#[tokio::test]
async fn postgres_migrations_create_the_full_schema() {
    let Some((_store, _guard)) = fresh_store().await else {
        return;
    };
    let db = _store.connection().await.expect("connection");

    for table in [
        "chat_threads",
        "chat_messages",
        "voice_profiles",
        "transcription_records",
        "runtime_jobs",
        "job_stages",
        "runtime_worker_heartbeats",
        "durable_idempotency_keys_v2",
        "gateway_principal_keys",
        "fleet_worker_observations",
        "fleet_capacity_claims",
    ] {
        let exists = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT 1 FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = ?1 LIMIT 1",
                    vec![table.into()],
                )
                .expect("table statement builds"),
            )
            .await
            .expect("table query runs");
        assert!(exists.is_some(), "{table} table exists on postgres");
    }

    // Epoch-milli timestamps overflow a 4-byte server INTEGER; the migrator
    // must have promoted them to BIGINT.
    for (table, column) in [
        ("chat_threads", "created_at"),
        ("fleet_capacity_claims", "expires_at"),
        ("runtime_jobs", "created_at"),
    ] {
        assert_eq!(
            column_type(db, table, column).await,
            "bigint",
            "{table}.{column} must be bigint on postgres"
        );
    }
    assert_eq!(
        column_type(db, "voice_turns", "audio_duration_secs").await,
        "double precision",
        "REAL columns must widen to double precision on postgres"
    );

    let profile = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT 1 FROM voice_profiles WHERE id = ?1 LIMIT 1",
                vec![crate::voice_defaults::DEFAULT_VOICE_PROFILE_ID.into()],
            )
            .expect("profile statement builds"),
        )
        .await
        .expect("profile query runs");
    assert!(profile.is_some(), "default voice profile is seeded");
}

#[tokio::test]
async fn postgres_fleet_observations_are_monotonic_per_incarnation() {
    let Some((store, _guard)) = fresh_store().await else {
        return;
    };
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
        store
            .observe_fleet_worker(&observation(2, "inc-1"))
            .await
            .unwrap(),
        "same-sequence re-publication refreshes the view"
    );

    // A replacement incarnation restarts its sequence counter and wins.
    assert!(store
        .observe_fleet_worker(&observation(1, "inc-2"))
        .await
        .unwrap());
    let views = store.read_fresh_fleet_workers(60_000).await.unwrap();
    assert_eq!(views.len(), 1);
    assert_eq!(views[0].incarnation_id, "inc-2");
    assert_eq!(views[0].status_sequence, 1);
}

#[tokio::test]
async fn postgres_fleet_capacity_claims_are_atomic_across_connections() {
    // Two independent store handles (separate pools, as two gateway
    // processes would hold) race claims against one shared database.
    let Some(url) = pg_url() else {
        return;
    };
    let _guard = PG_TEST_LOCK.lock().await;
    let mut stores = Vec::new();
    for _ in 0..2 {
        let mut store =
            BatchRuntimeStore::initialize_with_database(StoreDatabase::from_url(url.clone()));
        // Warm both connections (running migrations) before the race so the
        // test measures claim atomicity, not first-open contention.
        store.connection().await.expect("postgres connection opens");
        stores.push(store);
    }
    {
        let connection = stores[0].connection().await.expect("connection");
        drop_all_tables(connection).await.expect("tables drop");
        Migrator::up(connection).await.expect("migrations run");
    }

    let mut handles = Vec::new();
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
async fn postgres_fleet_claims_honor_ttl_owner_release_and_reap() {
    let Some((mut store, _guard)) = fresh_store().await else {
        return;
    };
    // Claims mint under the controlled clock so their expiry is testable.
    store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
    let claim = store
        .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-a", 2, 60_000)
        .await
        .unwrap()
        .expect("claim");
    assert!(
        !store
            .release_fleet_capacity(&claim, "gateway-b")
            .await
            .unwrap(),
        "a peer gateway must not release another gateway's claim"
    );

    store
        .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-b", 2, 60_000)
        .await
        .unwrap()
        .expect("second claim");
    assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 2);

    assert_eq!(store.release_gateway_claims("gateway-a").await.unwrap(), 1);
    assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 1);

    // Reader-side TTL: expired claims stop counting, then reap removes them.
    store.set_test_clock(Arc::new(AtomicI64::new(61_001)));
    assert_eq!(store.count_live_fleet_claims("worker-a").await.unwrap(), 0);
    assert_eq!(store.reap_expired_fleet_claims(64).await.unwrap(), 1);
    assert!(
        store
            .try_claim_fleet_capacity("worker-a", "inc-1", "gateway-c", 4, 60_000)
            .await
            .unwrap()
            .is_some(),
        "reaped capacity is reusable"
    );
}

#[tokio::test]
async fn postgres_durable_principal_keys_upsert_and_load() {
    let Some((store, _guard)) = fresh_store().await else {
        return;
    };
    let db = store.connection().await.expect("connection");
    for (principal, roles, tenant, salt, hash) in [
        (
            "principal-a",
            r#"["inference"]"#,
            "tenant-a",
            "a".repeat(32),
            "b".repeat(64),
        ),
        (
            "principal-b",
            r#"["admin","metrics"]"#,
            "tenant-b",
            "c".repeat(32),
            "d".repeat(64),
        ),
    ] {
        db.execute_raw(
            raw::statement(
                db,
                "INSERT INTO gateway_principal_keys (principal_id, roles_json, tenant_id, key_salt, key_hash, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
                vec![
                    principal.into(),
                    roles.into(),
                    tenant.into(),
                    salt.into(),
                    hash.into(),
                    1_758_900_000_000_i64.into(),
                ],
            )
            .expect("principal insert builds"),
        )
        .await
        .expect("principal insert runs");
    }

    // The ON CONFLICT upsert path replaces an existing principal in place.
    db.execute_raw(
        raw::statement(
            db,
            "INSERT INTO gateway_principal_keys (principal_id, roles_json, tenant_id, key_salt, key_hash, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) ON CONFLICT(principal_id) DO UPDATE SET roles_json = excluded.roles_json, updated_at = excluded.updated_at",
            vec![
                "principal-a".into(),
                r#"["inference","admin"]"#.into(),
                "tenant-a".into(),
                "a".repeat(32).into(),
                "b".repeat(64).into(),
                1_758_900_000_001_i64.into(),
            ],
        )
        .expect("principal upsert builds"),
    )
    .await
    .expect("principal upsert runs");

    let row = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT roles_json FROM gateway_principal_keys WHERE principal_id = ?1",
                vec!["principal-a".into()],
            )
            .expect("principal select builds"),
        )
        .await
        .expect("principal select runs")
        .expect("principal-a row");
    let roles_json: String = row.try_get_by_index(0).expect("roles_json value");
    assert_eq!(roles_json, r#"["inference","admin"]"#);
    let count = db
        .query_one_raw(
            raw::statement(db, "SELECT COUNT(*) FROM gateway_principal_keys", vec![])
                .expect("count statement builds"),
        )
        .await
        .expect("count query runs")
        .expect("count row");
    let total: i64 = count.try_get_by_index(0).expect("count value");
    assert_eq!(total, 2, "upsert replaced principal-a in place");
}

#[tokio::test]
async fn postgres_chat_threads_store_big_epoch_millis() {
    let Some((store, _guard)) = fresh_store().await else {
        return;
    };
    let db = store.connection().await.expect("connection");
    let now: i64 = 1_758_900_123_456; // far beyond the 4-byte INTEGER range
    db.execute_raw(
        raw::statement(
            db,
            "INSERT INTO chat_threads (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?3)",
            vec!["thread-pg-1".into(), "DS5".into(), now.into()],
        )
        .expect("thread insert builds"),
    )
    .await
    .expect("thread insert runs");
    let row = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT created_at, updated_at FROM chat_threads WHERE id = ?1",
                vec!["thread-pg-1".into()],
            )
            .expect("thread select builds"),
        )
        .await
        .expect("thread select runs")
        .expect("thread row");
    let created: i64 = row.try_get_by_index(0).expect("created_at value");
    assert_eq!(created, now, "epoch-millis timestamps round-trip exactly");
}

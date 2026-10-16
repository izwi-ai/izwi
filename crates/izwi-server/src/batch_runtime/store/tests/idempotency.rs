use super::*;

#[test]
fn durable_idempotency_digest_uses_canonical_json_object_order() {
    let first: serde_json::Value = serde_json::from_str(
        r#"{"text":"hello","options":{"voice":"a","speed":1},"parts":[1,2]}"#,
    )
    .expect("first request");
    let reordered: serde_json::Value = serde_json::from_str(
        r#"{"parts":[1,2],"options":{"speed":1,"voice":"a"},"text":"hello"}"#,
    )
    .expect("reordered request");
    let changed: serde_json::Value = serde_json::from_str(
        r#"{"parts":[2,1],"options":{"speed":1,"voice":"a"},"text":"hello"}"#,
    )
    .expect("changed request");

    assert_eq!(
        canonical_request_digest(&first).expect("first digest"),
        canonical_request_digest(&reordered).expect("reordered digest")
    );
    assert_ne!(
        canonical_request_digest(&first).expect("first digest"),
        canonical_request_digest(&changed).expect("changed digest")
    );
    assert!(canonical_request_digest(&json!(
        "x".repeat(MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES + 1)
    ))
    .is_err());

    let mut too_deep = serde_json::Value::Null;
    for _ in 0..=MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH {
        too_deep = serde_json::Value::Array(vec![too_deep]);
    }
    assert!(canonical_request_digest(&too_deep)
        .unwrap_err()
        .to_string()
        .contains("maximum nesting depth"));
}

#[tokio::test]
async fn durable_idempotency_is_tenant_and_operation_scoped_with_replay() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let job = create_test_job(&store, 0, 1).await;
    let request = durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 500);
    let reservation = match store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("reserve")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    let committed = store
        .commit_durable_idempotency(&reservation, &job.id, json!({"job_id": job.id}), 2_000)
        .await
        .expect("commit")
        .expect("owned reservation");
    assert_eq!(committed.runtime_job_id, job.id);
    assert_eq!(
        store
            .commit_durable_idempotency(
                &reservation,
                &job.id,
                json!({"job_id": job.id}),
                2_000,
            )
            .await
            .expect("idempotent commit"),
        Some(committed)
    );

    assert!(matches!(
        store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("replay"),
        DurableIdempotencyBegin::Replay(replay)
            if replay.runtime_job_id == job.id
                && replay.response_json == json!({"job_id": job.id})
    ));
    assert!(matches!(
        store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                request_digest: sha256_hex(b"different"),
                ..request.clone()
            })
            .await
            .expect("conflict"),
        DurableIdempotencyBegin::Conflict
    ));
    assert!(matches!(
        store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                tenant_scope: "tenant-b".to_string(),
                ..request.clone()
            })
            .await
            .expect("other tenant"),
        DurableIdempotencyBegin::Acquired(_)
    ));
    assert!(matches!(
        store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                operation: "job.export".to_string(),
                ..request
            })
            .await
            .expect("other operation"),
        DurableIdempotencyBegin::Acquired(_)
    ));
}

#[tokio::test]
async fn concurrent_same_key_reservations_never_double_acquire() {
    // Fleet regression: two gateways (or two retries) may race to reserve
    // the same idempotency key. The conflict-tolerant insert must let
    // exactly one caller acquire; the loser observes the winner's exact
    // state instead of creating duplicate work. This holds on SQLite via
    // the named admission lock and on PostgreSQL via the
    // ON CONFLICT/INSERT IGNORE key fence.
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let first_request =
        durable_idempotency_request("tenant-a", "job.create", "race-key", b"one", 500);
    let second_request = first_request.clone();
    let (first, second) = tokio::join!(
        store.reserve_durable_idempotency(first_request),
        store.reserve_durable_idempotency(second_request)
    );
    let first = first.expect("first concurrent reserve");
    let second = second.expect("second concurrent reserve");
    let acquired = usize::from(matches!(first, DurableIdempotencyBegin::Acquired(_)))
        + usize::from(matches!(second, DurableIdempotencyBegin::Acquired(_)));
    assert_eq!(
        acquired, 1,
        "exactly one concurrent caller may acquire, got {first:?} and {second:?}"
    );
    for outcome in [&first, &second] {
        match outcome {
            DurableIdempotencyBegin::Acquired(_)
            | DurableIdempotencyBegin::InProgress { .. } => {}
            other => panic!("loser must observe the winner, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn durable_idempotency_reservations_expire_and_fence_stale_commits() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let job = create_test_job(&store, 0, 1).await;
    let request = durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 100);
    let first = match store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("first reserve")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    assert!(matches!(
        store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("in progress"),
        DurableIdempotencyBegin::InProgress { expires_at: 1_100 }
    ));

    clock.store(1_100, Ordering::SeqCst);
    let second = match store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("reserve after expiry")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    assert_ne!(second.reservation_token, first.reservation_token);
    assert!(store
        .commit_durable_idempotency(&first, &job.id, json!({"job_id": job.id}), 100)
        .await
        .expect("stale commit")
        .is_none());
    store
        .commit_durable_idempotency(&second, &job.id, json!({"job_id": job.id}), 100)
        .await
        .expect("current commit")
        .expect("current owner");

    clock.store(1_200, Ordering::SeqCst);
    assert!(matches!(
        store
            .reserve_durable_idempotency(request)
            .await
            .expect("reserve after result expiry"),
        DurableIdempotencyBegin::Acquired(_)
    ));
}

#[tokio::test]
async fn durable_idempotency_concurrent_reservation_has_one_owner_and_fenced_release() {
    let (mut store, _root) = build_store().await;
    store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
    let request =
        durable_idempotency_request("tenant-a", "job.create", "concurrent-key", b"one", 500);
    let (first, second) = tokio::join!(
        store.reserve_durable_idempotency(request.clone()),
        store.reserve_durable_idempotency(request.clone()),
    );
    let outcomes = [
        first.expect("first reserve"),
        second.expect("second reserve"),
    ];
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::Acquired(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::InProgress { .. }))
            .count(),
        1
    );
    let reservation = outcomes
        .into_iter()
        .find_map(|outcome| match outcome {
            DurableIdempotencyBegin::Acquired(reservation) => Some(reservation),
            _ => None,
        })
        .expect("reservation owner");
    let mut forged = reservation.clone();
    forged.reservation_token = new_uuid();
    assert!(!store
        .release_durable_idempotency(&forged)
        .await
        .expect("forged release"));
    assert!(store
        .release_durable_idempotency(&reservation)
        .await
        .expect("owner release"));
    assert!(matches!(
        store
            .reserve_durable_idempotency(request)
            .await
            .expect("reserve after release"),
        DurableIdempotencyBegin::Acquired(_)
    ));
}

#[tokio::test]
async fn durable_idempotency_pruning_and_capacity_are_bounded_and_deterministic() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    for (offset, key) in [(0, "a"), (1, "b"), (2, "c")] {
        clock.store(1_000 + offset, Ordering::SeqCst);
        assert!(matches!(
            store
                .reserve_durable_idempotency(durable_idempotency_request(
                    "tenant-a",
                    "job.create",
                    key,
                    key.as_bytes(),
                    100,
                ))
                .await
                .expect("reserve expiring key"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }
    clock.store(1_200, Ordering::SeqCst);
    assert_eq!(
        store
            .prune_expired_durable_idempotency(2)
            .await
            .expect("bounded prune"),
        2
    );
    let db = store.connection().await.expect("database");
    let rows = db
        .query_all_raw(
            raw::statement(
                db,
                "SELECT idempotency_key FROM durable_idempotency_keys_v2 ORDER BY idempotency_key",
                vec![],
            )
            .expect("remaining-key query"),
        )
        .await
        .expect("remaining keys");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].try_get_by_index::<String>(0).unwrap(), "c");

    store
        .prune_expired_durable_idempotency(1)
        .await
        .expect("remove final expired key");
    clock.store(2_000, Ordering::SeqCst);
    for key in ["one", "two"] {
        assert!(matches!(
            store
                .reserve_durable_idempotency_with_capacity(
                    durable_idempotency_request(
                        "tenant-a",
                        "job.create",
                        key,
                        key.as_bytes(),
                        100,
                    ),
                    2,
                )
                .await
                .expect("within capacity"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }
    assert!(matches!(
        store
            .reserve_durable_idempotency_with_capacity(
                durable_idempotency_request("tenant-a", "job.create", "three", b"three", 100,),
                2,
            )
            .await
            .expect("capacity outcome"),
        DurableIdempotencyBegin::CapacityExceeded
    ));
}

/// DUR-F4: idempotency capacity is per tenant, with a global ceiling,
/// kept as counters by the transactions that insert and delete records.
#[tokio::test]
async fn durable_idempotency_quota_is_per_tenant_and_counted_incrementally() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let limits = QuotaLimits {
        global: 3,
        per_tenant: 2,
    };
    let acquired = |outcome: &DurableIdempotencyBegin| {
        matches!(outcome, DurableIdempotencyBegin::Acquired(_))
    };

    assert!(acquired(
        &reserve_with_limits(&store, "tenant-a", "a1", limits).await
    ));
    let DurableIdempotencyBegin::Acquired(a2) =
        reserve_with_limits(&store, "tenant-a", "a2", limits).await
    else {
        panic!("tenant-a has room for two records");
    };
    // A full tenant is refused without touching anyone else's capacity.
    assert_eq!(
        reserve_with_limits(&store, "tenant-a", "a3", limits).await,
        DurableIdempotencyBegin::CapacityExceeded
    );
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-b", "b1", limits).await
    ));
    // Duplicates neither count nor need capacity.
    assert!(matches!(
        reserve_with_limits(&store, "tenant-a", "a1", limits).await,
        DurableIdempotencyBegin::InProgress { .. }
    ));
    // The global ceiling holds across tenants.
    assert_eq!(
        reserve_with_limits(&store, "tenant-c", "c1", limits).await,
        DurableIdempotencyBegin::CapacityExceeded
    );
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;

    assert!(store.release_durable_idempotency(&a2).await.unwrap());
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-c", "c1", limits).await
    ));

    // A tenant at its cap first gives back its own expired records.
    clock.store(2_000, Ordering::SeqCst);
    let roomy = QuotaLimits {
        global: 16,
        per_tenant: 2,
    };
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-b", "b2", roomy).await
    ));
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-b", "b3", roomy).await
    ));
    assert_eq!(
        count_where(
            &store,
            "SELECT COUNT(*) FROM durable_idempotency_keys_v2 WHERE tenant_scope = 'tenant-a'",
            vec![]
        )
        .await,
        1,
        "another tenant's expired records wait for the sweeper"
    );
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;

    assert_eq!(
        store.prune_expired_durable_idempotency(64).await.unwrap(),
        2
    );
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;
}

/// An upgrade creates the quota table and seeds it from the records that
/// already exist; later migrations leave the live counts alone.
#[tokio::test]
async fn quota_usage_is_backfilled_once_when_its_table_is_created() {
    let (store, _root) = build_store().await;
    for (tenant, key) in [("tenant-a", "a1"), ("tenant-a", "a2"), ("tenant-b", "b1")] {
        assert!(matches!(
            store
                .reserve_durable_idempotency(durable_idempotency_request(
                    tenant,
                    "job.create",
                    key,
                    key.as_bytes(),
                    60_000,
                ))
                .await
                .unwrap(),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }
    let db = store.connection().await.unwrap();
    // A database from before the usage table and versioned migrations.
    db.execute_unprepared("DROP TABLE runtime_quota_usage")
        .await
        .unwrap();
    db.execute_unprepared("DROP TABLE schema_migrations")
        .await
        .unwrap();
    crate::db::migrator::Migrator::up(db).await.unwrap();
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;
    assert_eq!(
        count_where(
            &store,
            "SELECT used FROM runtime_quota_usage WHERE quota_kind = 'durable_idempotency' AND scope = ''",
            vec![]
        )
        .await,
        3
    );

    db.execute_unprepared(
        "UPDATE runtime_quota_usage SET used = 7 WHERE quota_kind = 'durable_idempotency' AND scope = ''",
    )
    .await
    .unwrap();
    crate::db::migrator::Migrator::up(db).await.unwrap();
    assert_eq!(
        count_where(
            &store,
            "SELECT used FROM runtime_quota_usage WHERE quota_kind = 'durable_idempotency' AND scope = ''",
            vec![]
        )
        .await,
        7,
        "an existing table is never re-seeded"
    );
}

#[tokio::test]
async fn durable_idempotency_rejects_unbounded_or_unattached_records() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let oversized_key = "x".repeat(MAX_DURABLE_IDEMPOTENCY_KEY_BYTES + 1);
    assert!(store
        .reserve_durable_idempotency(durable_idempotency_request(
            "tenant-a",
            "job.create",
            &oversized_key,
            b"one",
            100,
        ))
        .await
        .unwrap_err()
        .to_string()
        .contains("idempotency key exceeds"));
    assert!(store
        .reserve_durable_idempotency(DurableIdempotencyRequest {
            request_digest: "A".repeat(64),
            ..durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 100,)
        })
        .await
        .unwrap_err()
        .to_string()
        .contains("lowercase SHA-256"));

    let reservation = match store
        .reserve_durable_idempotency(durable_idempotency_request(
            "tenant-a",
            "job.create",
            "key-2",
            b"two",
            100,
        ))
        .await
        .expect("reserve")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    assert!(store
        .commit_durable_idempotency(
            &reservation,
            "missing-job",
            json!({"job_id": "missing-job"}),
            100,
        )
        .await
        .expect("unattached commit")
        .is_none());
    let job = create_test_job(&store, 0, 1).await;
    assert!(store
        .commit_durable_idempotency(
            &reservation,
            &job.id,
            json!({"result": "x".repeat(MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES)}),
            100,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("result exceeds"));
}

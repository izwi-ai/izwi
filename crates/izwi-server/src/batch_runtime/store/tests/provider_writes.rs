use super::*;

#[test]
fn provider_write_request_envelope_is_typed_and_bounded() {
    let mut input = test_provider_write_input(
        MediaNamespace::Other("artifact-store".to_string()),
        "artifact-store",
        1_000,
    );
    input
        .provider_request
        .metadata
        .insert("workflow_stage".to_string(), "finalize".to_string());
    validate_provider_write_input(&input).expect("valid exact provider request");
    let serialized = serialize_provider_write_request(&input.provider_request).unwrap();
    assert_eq!(
        deserialize_provider_write_request(&serialized).unwrap(),
        input.provider_request
    );
    assert!(deserialize_provider_write_request("{}").is_err());
    assert!(deserialize_provider_write_request(
        &"x".repeat(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES + 1)
    )
    .unwrap_err()
    .to_string()
    .contains("exceeds"));
    let unsupported = serialized.replacen("\"version\":1", "\"version\":2", 1);
    assert!(deserialize_provider_write_request(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("version is unsupported"));
    let mut nested_unknown: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    nested_unknown["request"]["future_field"] = json!("must-not-be-dropped");
    assert!(
        deserialize_provider_write_request(&nested_unknown.to_string())
            .unwrap_err()
            .to_string()
            .contains("invalid")
    );

    let mut invalid = input.clone();
    invalid.write_id = invalid.write_id.to_ascii_uppercase();
    invalid.provider_request.record_id = invalid.write_id.clone();
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("identity"));

    let mut invalid = input.clone();
    invalid.provider_request.record_id = new_uuid();
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("record identity"));

    let mut invalid = input.clone();
    invalid.provider_request.namespace = MediaNamespace::Other("other".to_string());
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("namespace does not match"));

    let mut invalid = input.clone();
    invalid.provider_request.namespace = MediaNamespace::Other(String::new());
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("request namespace"));

    let mut invalid = input.clone();
    invalid.provider_request.content_type = "text/plain".to_string();
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("does not match"));

    let mut invalid = input.clone();
    invalid.provider_request.metadata = (0..=MAX_PROVIDER_WRITE_METADATA_ENTRIES)
        .map(|index| (format!("key-{index}"), "value".to_string()))
        .collect();
    invalid
        .provider_request
        .metadata
        .insert("tenant_id".to_string(), "tenant-a".to_string());
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("entry count"));

    let mut invalid = input.clone();
    invalid.provider_request.metadata.insert(
        "k".repeat(MAX_PROVIDER_WRITE_METADATA_KEY_BYTES + 1),
        "value".to_string(),
    );
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("metadata key"));

    let mut invalid = input.clone();
    invalid.provider_request.metadata.insert(
        "large".to_string(),
        "v".repeat(MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES + 1),
    );
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("metadata value"));

    let mut invalid = input.clone();
    for index in 0..5 {
        invalid
            .provider_request
            .metadata
            .insert(format!("aggregate-{index}"), "v".repeat(900));
    }
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("aggregate byte limit"));

    let mut invalid = input.clone();
    invalid.provider_request.namespace =
        MediaNamespace::Other("n".repeat(MAX_PROVIDER_WRITE_NAMESPACE_BYTES + 1));
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("request namespace"));

    let mut invalid = input;
    invalid.provider_request.metadata.clear();
    invalid
        .provider_request
        .metadata
        .insert("tenant_id".to_string(), "tenant-a".to_string());
    for index in 0..4 {
        invalid
            .provider_request
            .metadata
            .insert(format!("escaped-{index}"), "\\\"".repeat(500));
    }
    assert!(serialize_provider_write_request(&invalid.provider_request)
        .unwrap_err()
        .to_string()
        .contains("envelope exceeds"));
}

#[tokio::test]
async fn provider_write_request_roundtrips_and_legacy_recovery_fails_closed() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());

    let mut exact =
        test_provider_write_input(MediaNamespace::GeneratedSpeech, "generated_speech", 10);
    exact
        .provider_request
        .metadata
        .insert("workflow_stage".to_string(), "finalize".to_string());
    let exact = store.reserve_provider_write(exact).await.unwrap();
    assert_eq!(
        exact.provider_request.namespace,
        MediaNamespace::GeneratedSpeech
    );
    assert_eq!(
        exact
            .provider_request
            .metadata
            .get("workflow_stage")
            .map(String::as_str),
        Some("finalize")
    );

    let legacy = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            10,
        ))
        .await
        .unwrap();
    let db = store.connection().await.unwrap();
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE provider_write_reservations SET provider_request_json = NULL WHERE write_id IN (?1, ?2)",
            vec![exact.write_id.clone().into(), legacy.write_id.clone().into()],
        )
        .unwrap(),
    )
    .await
    .unwrap();

    clock.store(1_010, Ordering::SeqCst);
    let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].write_id, legacy.write_id);
    assert_eq!(
        claimed[0].provider_request,
        legacy_provider_write_request(
            &legacy.write_id,
            &legacy.tenant_scope,
            &legacy.storage_namespace,
            &legacy.content_type,
            legacy.filename.as_deref(),
        )
        .unwrap()
    );
    let retained = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = ?1 AND provider_request_json IS NULL",
                vec![exact.write_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(retained, 1);
}

#[tokio::test]
async fn invalid_provider_write_rows_remain_fenced_without_starving_cleanup() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let poison = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            10,
        ))
        .await
        .unwrap();
    let oversized = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::GeneratedSpeech,
            "generated_speech",
            10,
        ))
        .await
        .unwrap();
    let valid = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            10,
        ))
        .await
        .unwrap();
    let db = store.connection().await.unwrap();
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE provider_write_reservations SET provider_request_json = ?1, created_at = 0 WHERE write_id = ?2",
            vec![
                "{\"version\":1,\"request\":{\"future_field\":true}}"
                    .into(),
                poison.write_id.clone().into(),
            ],
        )
        .unwrap(),
    )
    .await
    .unwrap();
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE provider_write_reservations SET provider_request_json = ?1, created_at = -1 WHERE write_id = ?2",
            vec![
                "x"
                    .repeat(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES + 1)
                    .into(),
                oversized.write_id.clone().into(),
            ],
        )
        .unwrap(),
    )
    .await
    .unwrap();

    clock.store(1_010, Ordering::SeqCst);
    let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].write_id, valid.write_id);
    assert!(get_provider_write_with(db, &oversized.write_id)
        .await
        .unwrap()
        .is_none());

    let quarantined = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT state, available_at, cleanup_attempt_count, last_error FROM provider_write_reservations WHERE write_id = ?1",
                vec![poison.write_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        quarantined.try_get_by_index::<String>(0).unwrap(),
        "cleanup_pending"
    );
    assert!(quarantined.try_get_by_index::<i64>(1).unwrap() > 1_010);
    assert_eq!(quarantined.try_get_by_index::<i64>(2).unwrap(), 1);
    assert!(quarantined
        .try_get_by_index::<Option<String>>(3)
        .unwrap()
        .is_some());

    let oversized_retained = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = ?1 AND state = 'reserved'",
                vec![oversized.write_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(oversized_retained, 1);
}

/// DUR-F4: provider write reservations are capped per tenant and given
/// back by every path that deletes them.
#[tokio::test]
async fn provider_write_quota_is_per_tenant_and_released_by_cleanup() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    store.set_provider_write_limits_for_test(QuotaLimits {
        global: 2,
        per_tenant: 1,
    });
    let for_tenant = |tenant: &str| {
        let mut input = test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            100,
        );
        input.tenant_scope = tenant.to_string();
        input
            .provider_request
            .metadata
            .insert("tenant_id".to_string(), tenant.to_string());
        input
    };

    let first = store
        .reserve_provider_write(for_tenant("tenant-a"))
        .await
        .expect("first reservation");
    let error = store
        .reserve_provider_write(for_tenant("tenant-a"))
        .await
        .expect_err("tenant-a is at its cap");
    assert!(error.to_string().contains("capacity exhausted"));
    store
        .reserve_provider_write(for_tenant("tenant-b"))
        .await
        .expect("tenant-b has its own capacity");
    assert!(store
        .reserve_provider_write(for_tenant("tenant-c"))
        .await
        .is_err());
    assert_quota_matches_records(
        &store,
        QuotaKind::ProviderWrite,
        "provider_write_reservations",
    )
    .await;

    assert!(store
        .abandon_provider_write(&first, "test abandon")
        .await
        .unwrap());
    clock.store(2_000, Ordering::SeqCst);
    let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(claimed.len(), 2);
    for reservation in &claimed {
        assert!(store
            .complete_provider_write_cleanup(reservation)
            .await
            .unwrap());
    }
    assert_quota_matches_records(
        &store,
        QuotaKind::ProviderWrite,
        "provider_write_reservations",
    )
    .await;
    store
        .reserve_provider_write(for_tenant("tenant-a"))
        .await
        .expect("capacity returns after cleanup");
}

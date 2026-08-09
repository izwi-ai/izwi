//! Downstream-facing compile and serialized-configuration compatibility fixture.

use izwi_core::backends::BackendPreference;
use izwi_core::config::{EngineConfig, KvCacheDtype, PrefixCachePolicy};
use izwi_core::ManagedKvRuntimeSnapshot;

const BETA17_CONFIG: &str = include_str!("fixtures/engine-config-beta17.json");
const BETA18_CONFIG: &str = include_str!("fixtures/engine-config-beta18.json");

#[test]
fn beta17_config_loads_with_safe_kv_defaults() {
    let config: EngineConfig = serde_json::from_str(BETA17_CONFIG).unwrap();
    assert_eq!(config.kv_page_size, 64);
    assert_eq!(config.backend, BackendPreference::Auto);
    assert!(!config.enable_prefix_caching);
    assert!(config.managed_prefix_cache_salt.is_none());

    let policy = config.resolved_kv_cache_policy(256).unwrap();
    assert_eq!(policy.requested.page_size, 64);
    assert_eq!(policy.effective.page_size, 64);
    assert_eq!(policy.effective.dtype, KvCacheDtype::Float16);
    assert_eq!(policy.effective.prefix, PrefixCachePolicy::Disabled);
    assert!(policy.fallback_reason.is_none());
}

#[test]
fn beta18_config_reports_requested_and_effective_prefix_capacity() {
    let config: EngineConfig = serde_json::from_str(BETA18_CONFIG).unwrap();
    let policy = config.resolved_kv_cache_policy(40).unwrap();

    assert_eq!(policy.requested.page_size, 64);
    assert_eq!(policy.effective.page_size, 64);
    assert_eq!(policy.requested.dtype, KvCacheDtype::Float16);
    assert_eq!(policy.effective.dtype, KvCacheDtype::Float16);
    assert_eq!(
        policy.requested.prefix,
        PrefixCachePolicy::Namespaced {
            namespace: "deployment-a/tenant-42".into(),
            max_pages: 24,
        }
    );
    assert_eq!(
        policy.effective.prefix,
        PrefixCachePolicy::Namespaced {
            namespace: "deployment-a/tenant-42".into(),
            max_pages: 8,
        }
    );
    assert!(policy
        .fallback_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("clamped from 24 to 8")));
}

#[test]
fn public_kv_observability_types_remain_constructible_and_serializable() {
    let snapshot = ManagedKvRuntimeSnapshot::default();
    let json = serde_json::to_value(snapshot).unwrap();

    assert_eq!(
        json["memory_accounting"],
        "resident_paged_plus_authorized_tensor"
    );
    assert!(json["totals"].is_object());
    assert!(json["counters"].is_object());
    assert!(json["models"].is_array());
}

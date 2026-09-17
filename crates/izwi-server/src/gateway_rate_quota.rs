//! Bounded, single-gateway tenant request-rate enforcement.
//!
//! This deliberately accounts only for request starts. Concurrent-work permits
//! need a longer lifetime than the public response when worker teardown is not
//! yet confirmed, so that separate Phase 5 safeguard is not modeled here.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const NANOS_PER_MINUTE: u128 = 60_000_000_000;
const MAX_ENV_VALUE_BYTES: usize = 20;

pub const DEFAULT_GATEWAY_TENANT_REQUESTS_PER_MINUTE: u32 = 600;
pub const DEFAULT_GATEWAY_TENANT_BURST_REQUESTS: u32 = 64;
pub const DEFAULT_GATEWAY_MAX_TRACKED_TENANTS: usize = 16_384;
pub const MAX_GATEWAY_TENANT_REQUESTS_PER_MINUTE: u32 = 1_000_000;
pub const MAX_GATEWAY_TENANT_BURST_REQUESTS: u32 = 100_000;
pub const MAX_GATEWAY_TRACKED_TENANTS: usize = 65_536;

const REQUESTS_PER_MINUTE_ENV: &str = "IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE";
const BURST_REQUESTS_ENV: &str = "IZWI_GATEWAY_TENANT_BURST_REQUESTS";
const MAX_TRACKED_TENANTS_ENV: &str = "IZWI_GATEWAY_MAX_TRACKED_TENANTS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Validated process-local tenant request-rate policy.
///
/// Production overrides use `IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE`,
/// `IZWI_GATEWAY_TENANT_BURST_REQUESTS`, and
/// `IZWI_GATEWAY_MAX_TRACKED_TENANTS`. These limits are not a multi-gateway
/// quota authority: fleet deployments must use shared or partitioned budgets.
pub struct GatewayRateQuotaConfig {
    requests_per_minute: u32,
    burst_requests: u32,
    max_tracked_tenants: usize,
}

impl GatewayRateQuotaConfig {
    pub fn requests_per_minute(&self) -> u32 {
        self.requests_per_minute
    }
    pub fn burst_requests(&self) -> u32 {
        self.burst_requests
    }
    pub fn max_tracked_tenants(&self) -> usize {
        self.max_tracked_tenants
    }
    pub fn new(
        requests_per_minute: u32,
        burst_requests: u32,
        max_tracked_tenants: usize,
    ) -> Result<Self, GatewayRateQuotaConfigError> {
        if !(1..=MAX_GATEWAY_TENANT_REQUESTS_PER_MINUTE).contains(&requests_per_minute) {
            return Err(GatewayRateQuotaConfigError::RequestsPerMinute);
        }
        if !(1..=MAX_GATEWAY_TENANT_BURST_REQUESTS).contains(&burst_requests) {
            return Err(GatewayRateQuotaConfigError::BurstRequests);
        }
        if !(1..=MAX_GATEWAY_TRACKED_TENANTS).contains(&max_tracked_tenants) {
            return Err(GatewayRateQuotaConfigError::TrackedTenants);
        }
        Ok(Self {
            requests_per_minute,
            burst_requests,
            max_tracked_tenants,
        })
    }

    pub fn from_env() -> Result<Self, GatewayRateQuotaConfigError> {
        Self::new(
            bounded_integer_env(
                REQUESTS_PER_MINUTE_ENV,
                DEFAULT_GATEWAY_TENANT_REQUESTS_PER_MINUTE,
            )?,
            bounded_integer_env(BURST_REQUESTS_ENV, DEFAULT_GATEWAY_TENANT_BURST_REQUESTS)?,
            bounded_integer_env(MAX_TRACKED_TENANTS_ENV, DEFAULT_GATEWAY_MAX_TRACKED_TENANTS)?,
        )
    }
}

impl Default for GatewayRateQuotaConfig {
    fn default() -> Self {
        Self::new(
            DEFAULT_GATEWAY_TENANT_REQUESTS_PER_MINUTE,
            DEFAULT_GATEWAY_TENANT_BURST_REQUESTS,
            DEFAULT_GATEWAY_MAX_TRACKED_TENANTS,
        )
        .expect("gateway rate-quota defaults must remain valid")
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GatewayRateQuotaConfigError {
    #[error("IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE must be a bounded non-zero integer")]
    RequestsPerMinute,
    #[error("IZWI_GATEWAY_TENANT_BURST_REQUESTS must be a bounded non-zero integer")]
    BurstRequests,
    #[error("IZWI_GATEWAY_MAX_TRACKED_TENANTS must be a bounded non-zero integer")]
    TrackedTenants,
}

fn bounded_integer_env<T>(name: &str, default: T) -> Result<T, GatewayRateQuotaConfigError>
where
    T: std::str::FromStr,
{
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default);
    };
    let raw = raw.to_str().filter(|value| {
        !value.is_empty()
            && value.len() <= MAX_ENV_VALUE_BYTES
            && value.bytes().all(|byte| byte.is_ascii_digit())
    });
    raw.and_then(|value| value.parse().ok())
        .ok_or_else(|| match name {
            REQUESTS_PER_MINUTE_ENV => GatewayRateQuotaConfigError::RequestsPerMinute,
            BURST_REQUESTS_ENV => GatewayRateQuotaConfigError::BurstRequests,
            _ => GatewayRateQuotaConfigError::TrackedTenants,
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GatewayRateDecision {
    Allowed,
    Limited,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GatewayRateQuotaError {
    #[error("gateway tenant rate state is unavailable")]
    StateUnavailable,
}

#[derive(Clone)]
pub(crate) struct GatewayRateQuota {
    inner: Arc<GatewayRateQuotaInner>,
}

struct GatewayRateQuotaInner {
    config: GatewayRateQuotaConfig,
    started_at: Instant,
    state: Mutex<RateState>,
}

#[derive(Default)]
struct RateState {
    tenants: BTreeMap<[u8; 32], TenantBucket>,
    access_sequence: u64,
}

struct TenantBucket {
    available_units: u128,
    last_refill: Duration,
    last_access_sequence: u64,
}

impl GatewayRateQuota {
    pub(crate) fn new(config: GatewayRateQuotaConfig) -> Self {
        Self {
            inner: Arc::new(GatewayRateQuotaInner {
                config,
                started_at: Instant::now(),
                state: Mutex::new(RateState::default()),
            }),
        }
    }

    pub(crate) fn check(
        &self,
        tenant_key: [u8; 32],
    ) -> Result<GatewayRateDecision, GatewayRateQuotaError> {
        self.check_at(tenant_key, self.inner.started_at.elapsed())
    }

    fn check_at(
        &self,
        tenant_key: [u8; 32],
        now: Duration,
    ) -> Result<GatewayRateDecision, GatewayRateQuotaError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| GatewayRateQuotaError::StateUnavailable)?;
        state.access_sequence = state.access_sequence.saturating_add(1);
        let access_sequence = state.access_sequence;

        let capacity = u128::from(self.inner.config.burst_requests) * NANOS_PER_MINUTE;
        if !state.tenants.contains_key(&tenant_key)
            && state.tenants.len() == self.inner.config.max_tracked_tenants
        {
            // Only a fully replenished bucket is inactive and safe to forget.
            // If all tracked tenants still carry rate-limit history, fail
            // closed instead of letting identity churn reset their budgets.
            let evicted = state
                .tenants
                .iter()
                .filter(|(_, bucket)| {
                    replenished_units(bucket, now, self.inner.config.requests_per_minute)
                        >= capacity
                })
                .min_by_key(|(key, bucket)| (bucket.last_access_sequence, *key))
                .map(|(key, _)| *key);
            let Some(evicted) = evicted else {
                return Ok(GatewayRateDecision::Limited);
            };
            state.tenants.remove(&evicted);
        }

        let bucket = state.tenants.entry(tenant_key).or_insert(TenantBucket {
            available_units: capacity,
            last_refill: now,
            last_access_sequence: access_sequence,
        });
        bucket.available_units =
            replenished_units(bucket, now, self.inner.config.requests_per_minute).min(capacity);
        bucket.last_refill = now;
        bucket.last_access_sequence = access_sequence;

        if bucket.available_units < NANOS_PER_MINUTE {
            return Ok(GatewayRateDecision::Limited);
        }
        bucket.available_units -= NANOS_PER_MINUTE;
        Ok(GatewayRateDecision::Allowed)
    }

    #[cfg(test)]
    fn tracked_tenants(&self) -> Result<usize, GatewayRateQuotaError> {
        self.inner
            .state
            .lock()
            .map(|state| state.tenants.len())
            .map_err(|_| GatewayRateQuotaError::StateUnavailable)
    }
}

fn replenished_units(bucket: &TenantBucket, now: Duration, requests_per_minute: u32) -> u128 {
    let elapsed_nanos = now.saturating_sub(bucket.last_refill).as_nanos();
    bucket
        .available_units
        .saturating_add(elapsed_nanos.saturating_mul(u128::from(requests_per_minute)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: u8) -> [u8; 32] {
        [value; 32]
    }

    #[test]
    fn token_bucket_is_deterministic_and_refills_without_refunding_denials() {
        let quota = GatewayRateQuota::new(GatewayRateQuotaConfig::new(60, 2, 4).unwrap());
        let start = Duration::ZERO;
        assert_eq!(
            quota.check_at(key(1), start).unwrap(),
            GatewayRateDecision::Allowed
        );
        assert_eq!(
            quota.check_at(key(1), start).unwrap(),
            GatewayRateDecision::Allowed
        );
        assert_eq!(
            quota.check_at(key(1), start).unwrap(),
            GatewayRateDecision::Limited
        );
        assert_eq!(
            quota.check_at(key(1), Duration::from_millis(999)).unwrap(),
            GatewayRateDecision::Limited
        );
        assert_eq!(
            quota.check_at(key(1), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Allowed
        );
        assert_eq!(
            quota.check_at(key(1), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Limited
        );
    }

    #[test]
    fn tenant_state_is_hard_bounded_and_evicts_least_recently_used() {
        let quota = GatewayRateQuota::new(GatewayRateQuotaConfig::new(60, 1, 2).unwrap());
        assert_eq!(
            quota.check_at(key(1), Duration::ZERO).unwrap(),
            GatewayRateDecision::Allowed
        );
        assert_eq!(
            quota.check_at(key(2), Duration::ZERO).unwrap(),
            GatewayRateDecision::Allowed
        );
        assert_eq!(
            quota.check_at(key(1), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Allowed,
            "tenant 1 becomes the active most-recent entry"
        );
        assert_eq!(
            quota.check_at(key(3), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Allowed,
            "fully replenished tenant 2 is safe to evict"
        );
        assert_eq!(quota.tracked_tenants().unwrap(), 2);
        assert_eq!(
            quota.check_at(key(2), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Limited,
            "identity churn must fail closed while all tracked buckets are active"
        );
        assert_eq!(quota.tracked_tenants().unwrap(), 2);
        assert_eq!(
            quota.check_at(key(2), Duration::from_secs(2)).unwrap(),
            GatewayRateDecision::Allowed,
            "an inactive bucket can be reclaimed deterministically"
        );
    }

    #[test]
    fn config_rejects_values_outside_hard_bounds() {
        assert_eq!(
            GatewayRateQuotaConfig::new(0, 1, 1),
            Err(GatewayRateQuotaConfigError::RequestsPerMinute)
        );
        assert_eq!(
            GatewayRateQuotaConfig::new(1, MAX_GATEWAY_TENANT_BURST_REQUESTS + 1, 1),
            Err(GatewayRateQuotaConfigError::BurstRequests)
        );
        assert_eq!(
            GatewayRateQuotaConfig::new(1, 1, MAX_GATEWAY_TRACKED_TENANTS + 1),
            Err(GatewayRateQuotaConfigError::TrackedTenants)
        );
    }
}

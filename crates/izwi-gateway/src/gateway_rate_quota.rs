//! Bounded, single-gateway tenant request-rate and token-rate enforcement.
//!
//! Requests are counted at their start. Tokens (prompt plus completion) are
//! reserved from an estimate at admission and settled to the actual usage
//! when the request completes. Concurrent-work permits need a longer lifetime
//! than the public response when worker teardown is not yet confirmed, so
//! that separate safeguard is not modeled here.

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
pub const MAX_GATEWAY_TENANT_TOKENS_PER_MINUTE: u32 = 1_000_000_000;

const REQUESTS_PER_MINUTE_ENV: &str = "IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE";
const BURST_REQUESTS_ENV: &str = "IZWI_GATEWAY_TENANT_BURST_REQUESTS";
const MAX_TRACKED_TENANTS_ENV: &str = "IZWI_GATEWAY_MAX_TRACKED_TENANTS";
const TOKENS_PER_MINUTE_ENV: &str = "IZWI_GATEWAY_TENANT_TOKENS_PER_MINUTE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Validated process-local tenant request-rate policy.
///
/// Production overrides use `IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE`,
/// `IZWI_GATEWAY_TENANT_BURST_REQUESTS`, `IZWI_GATEWAY_MAX_TRACKED_TENANTS`
/// and `IZWI_GATEWAY_TENANT_TOKENS_PER_MINUTE` (0 or unset: no token limit).
/// These limits are not a multi-gateway quota authority: fleet deployments
/// must use shared or partitioned budgets.
pub struct GatewayRateQuotaConfig {
    requests_per_minute: u32,
    burst_requests: u32,
    max_tracked_tenants: usize,
    tokens_per_minute: Option<u32>,
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
    pub fn tokens_per_minute(&self) -> Option<u32> {
        self.tokens_per_minute
    }
    /// Adds a per-tenant tokens-per-minute limit; `None` or 0 disables it.
    pub fn with_tokens_per_minute(
        mut self,
        tokens_per_minute: Option<u32>,
    ) -> Result<Self, GatewayRateQuotaConfigError> {
        let tokens_per_minute = tokens_per_minute.filter(|limit| *limit > 0);
        if tokens_per_minute.is_some_and(|limit| limit > MAX_GATEWAY_TENANT_TOKENS_PER_MINUTE) {
            return Err(GatewayRateQuotaConfigError::TokensPerMinute);
        }
        self.tokens_per_minute = tokens_per_minute;
        Ok(self)
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
            tokens_per_minute: None,
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
        )?
        .with_tokens_per_minute(Some(bounded_integer_env(TOKENS_PER_MINUTE_ENV, 0)?))
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
    #[error("IZWI_GATEWAY_TENANT_TOKENS_PER_MINUTE must be a bounded integer")]
    TokensPerMinute,
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
    raw.and_then(|value| value.parse().ok()).ok_or(match name {
        REQUESTS_PER_MINUTE_ENV => GatewayRateQuotaConfigError::RequestsPerMinute,
        BURST_REQUESTS_ENV => GatewayRateQuotaConfigError::BurstRequests,
        TOKENS_PER_MINUTE_ENV => GatewayRateQuotaConfigError::TokensPerMinute,
        _ => GatewayRateQuotaConfigError::TrackedTenants,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayRateDecision {
    Allowed,
    Limited,
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayRateQuotaError {
    #[error("gateway tenant rate state is unavailable")]
    StateUnavailable,
}

#[derive(Clone)]
pub struct GatewayRateQuota {
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
    /// Token balances, only with a tokens-per-minute limit. A balance may go
    /// negative when actual usage exceeds the estimate: the tenant then
    /// waits until it is repaid.
    tokens: BTreeMap<[u8; 32], TokenBucket>,
}

struct TokenBucket {
    /// Tokens in units of 1/`NANOS_PER_MINUTE` token.
    balance: i128,
    last_refill: Duration,
}

/// One limit's state for `x-ratelimit-*` headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitWindow {
    pub limit: u64,
    pub remaining: u64,
    /// Until the bucket is full again.
    pub reset: Duration,
    /// Until one more unit is available (zero when one already is).
    pub retry_after: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitSnapshot {
    pub requests: LimitWindow,
    pub tokens: Option<LimitWindow>,
}

/// The token-rate outcome of an admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenDecision {
    Allowed,
    Limited { retry_after: Duration },
}

struct TenantBucket {
    available_units: u128,
    last_refill: Duration,
    last_access_sequence: u64,
}

impl GatewayRateQuota {
    pub fn new(config: GatewayRateQuotaConfig) -> Self {
        Self {
            inner: Arc::new(GatewayRateQuotaInner {
                config,
                started_at: Instant::now(),
                state: Mutex::new(RateState::default()),
            }),
        }
    }

    pub fn check(
        &self,
        tenant_key: [u8; 32],
    ) -> Result<GatewayRateDecision, GatewayRateQuotaError> {
        self.check_at(tenant_key, self.inner.started_at.elapsed())
    }

    /// Whether `check` would allow the tenant now, without spending a token.
    pub fn peek(&self, tenant_key: [u8; 32]) -> Result<GatewayRateDecision, GatewayRateQuotaError> {
        let now = self.inner.started_at.elapsed();
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| GatewayRateQuotaError::StateUnavailable)?;
        let Some(bucket) = state.tenants.get(&tenant_key) else {
            return Ok(GatewayRateDecision::Allowed);
        };
        let capacity = u128::from(self.inner.config.burst_requests) * NANOS_PER_MINUTE;
        let available =
            replenished_units(bucket, now, self.inner.config.requests_per_minute).min(capacity);
        Ok(if available < NANOS_PER_MINUTE {
            GatewayRateDecision::Limited
        } else {
            GatewayRateDecision::Allowed
        })
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

    /// Reserves `estimate` tokens for an admitted request. Admits while the
    /// balance covers the estimate, capped at one minute's budget so a single
    /// large request can still run from a full bucket; the balance may then
    /// go negative. Without a token limit every request is allowed and
    /// nothing is reserved.
    pub fn reserve_tokens(
        &self,
        tenant_key: [u8; 32],
        estimate: u64,
    ) -> Result<(TokenDecision, Option<TokenReservation>), GatewayRateQuotaError> {
        let Some(tokens_per_minute) = self.inner.config.tokens_per_minute else {
            return Ok((TokenDecision::Allowed, None));
        };
        let now = self.inner.started_at.elapsed();
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| GatewayRateQuotaError::StateUnavailable)?;
        let capacity = i128::from(tokens_per_minute) * NANOS_PER_MINUTE as i128;
        if !state.tokens.contains_key(&tenant_key)
            && state.tokens.len() == self.inner.config.max_tracked_tenants
        {
            let evicted = state
                .tokens
                .iter()
                .filter(|(_, bucket)| refilled_balance(bucket, now, tokens_per_minute) >= capacity)
                .map(|(key, _)| *key)
                .next();
            let Some(evicted) = evicted else {
                return Ok((
                    TokenDecision::Limited {
                        retry_after: Duration::from_secs(1),
                    },
                    None,
                ));
            };
            state.tokens.remove(&evicted);
        }
        let bucket = state.tokens.entry(tenant_key).or_insert(TokenBucket {
            balance: capacity,
            last_refill: now,
        });
        bucket.balance = refilled_balance(bucket, now, tokens_per_minute).min(capacity);
        bucket.last_refill = now;
        let reserved = i128::from(estimate) * NANOS_PER_MINUTE as i128;
        let needed = reserved.min(capacity);
        if bucket.balance < needed {
            let missing = (needed - bucket.balance) as u128;
            return Ok((
                TokenDecision::Limited {
                    retry_after: units_duration(missing, tokens_per_minute),
                },
                None,
            ));
        }
        bucket.balance -= reserved;
        Ok((
            TokenDecision::Allowed,
            Some(TokenReservation {
                quota: self.clone(),
                tenant_key,
                reserved: estimate,
                keep_on_drop: false,
            }),
        ))
    }

    fn adjust_tokens(&self, tenant_key: [u8; 32], tokens: i128) {
        let Some(tokens_per_minute) = self.inner.config.tokens_per_minute else {
            return;
        };
        let Ok(mut state) = self.inner.state.lock() else {
            return;
        };
        if let Some(bucket) = state.tokens.get_mut(&tenant_key) {
            let capacity = i128::from(tokens_per_minute) * NANOS_PER_MINUTE as i128;
            bucket.balance = bucket
                .balance
                .saturating_add(tokens.saturating_mul(NANOS_PER_MINUTE as i128))
                .min(capacity);
        }
    }

    /// The tenant's request and token limits now, for `x-ratelimit-*`
    /// headers. Nothing is spent.
    pub fn snapshot(&self, tenant_key: [u8; 32]) -> Option<RateLimitSnapshot> {
        let now = self.inner.started_at.elapsed();
        let state = self.inner.state.lock().ok()?;
        let config = &self.inner.config;
        let request_capacity = u128::from(config.burst_requests) * NANOS_PER_MINUTE;
        let request_units = state
            .tenants
            .get(&tenant_key)
            .map_or(request_capacity, |bucket| {
                replenished_units(bucket, now, config.requests_per_minute).min(request_capacity)
            });
        let requests = LimitWindow {
            limit: u64::from(config.requests_per_minute),
            remaining: u64::try_from(request_units / NANOS_PER_MINUTE).unwrap_or(u64::MAX),
            reset: units_duration(request_capacity - request_units, config.requests_per_minute),
            retry_after: units_duration(
                NANOS_PER_MINUTE.saturating_sub(request_units),
                config.requests_per_minute,
            ),
        };
        let tokens = config.tokens_per_minute.map(|tokens_per_minute| {
            let capacity = i128::from(tokens_per_minute) * NANOS_PER_MINUTE as i128;
            let balance = state.tokens.get(&tenant_key).map_or(capacity, |bucket| {
                refilled_balance(bucket, now, tokens_per_minute).min(capacity)
            });
            LimitWindow {
                limit: u64::from(tokens_per_minute),
                remaining: u64::try_from(balance.max(0) / NANOS_PER_MINUTE as i128)
                    .unwrap_or(u64::MAX),
                reset: units_duration((capacity - balance) as u128, tokens_per_minute),
                retry_after: units_duration(
                    (NANOS_PER_MINUTE as i128 - balance).max(0) as u128,
                    tokens_per_minute,
                ),
            }
        });
        Some(RateLimitSnapshot { requests, tokens })
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

/// Tokens reserved for one admitted request. Settled to the actual usage on
/// completion; dropped unsettled, the reservation is refunded unless the
/// request already started streaming (then the estimate stands, so leaving
/// mid-stream cannot dodge the charge).
pub struct TokenReservation {
    quota: GatewayRateQuota,
    tenant_key: [u8; 32],
    reserved: u64,
    keep_on_drop: bool,
}

impl TokenReservation {
    pub fn mark_started(&mut self) {
        self.keep_on_drop = true;
    }

    pub fn settle(mut self, actual: u64) {
        self.quota.adjust_tokens(
            self.tenant_key,
            i128::from(self.reserved) - i128::from(actual),
        );
        self.keep_on_drop = true;
    }
}

impl Drop for TokenReservation {
    fn drop(&mut self) {
        if !self.keep_on_drop {
            self.quota
                .adjust_tokens(self.tenant_key, i128::from(self.reserved));
        }
    }
}

fn refilled_balance(bucket: &TokenBucket, now: Duration, tokens_per_minute: u32) -> i128 {
    let elapsed = now.saturating_sub(bucket.last_refill).as_nanos();
    let refill =
        i128::try_from(elapsed.saturating_mul(u128::from(tokens_per_minute))).unwrap_or(i128::MAX);
    bucket.balance.saturating_add(refill)
}

/// How long `units` (in 1/`NANOS_PER_MINUTE` steps) take to refill at
/// `per_minute`.
fn units_duration(units: u128, per_minute: u32) -> Duration {
    let nanos = units.div_ceil(u128::from(per_minute.max(1)));
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
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
    fn exhausted_tenant_does_not_affect_other_tenants() {
        let quota = GatewayRateQuota::new(GatewayRateQuotaConfig::new(60, 1, 4).unwrap());
        assert_eq!(
            quota.check_at(key(1), Duration::ZERO).unwrap(),
            GatewayRateDecision::Allowed
        );
        assert_eq!(
            quota.check_at(key(1), Duration::ZERO).unwrap(),
            GatewayRateDecision::Limited,
            "tenant 1 exhausts its own burst"
        );
        assert_eq!(
            quota.check_at(key(2), Duration::ZERO).unwrap(),
            GatewayRateDecision::Allowed,
            "tenant 2 is unaffected by tenant 1's exhaustion"
        );
        assert_eq!(
            quota.check_at(key(2), Duration::ZERO).unwrap(),
            GatewayRateDecision::Limited
        );
        assert_eq!(
            quota.check_at(key(2), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Allowed,
            "tenants refill on their own clocks"
        );
        assert_eq!(
            quota.check_at(key(1), Duration::from_secs(1)).unwrap(),
            GatewayRateDecision::Allowed
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
    fn token_reservations_settle_to_actual_usage_and_refund_when_dropped() {
        let quota = GatewayRateQuota::new(
            GatewayRateQuotaConfig::new(600, 64, 8)
                .unwrap()
                .with_tokens_per_minute(Some(100))
                .unwrap(),
        );
        let tenant = key(1);
        let remaining =
            |quota: &GatewayRateQuota| quota.snapshot(tenant).unwrap().tokens.unwrap().remaining;

        let (decision, reservation) = quota.reserve_tokens(tenant, 40).unwrap();
        assert_eq!(decision, TokenDecision::Allowed);
        assert!((60..=61).contains(&remaining(&quota)));
        reservation.unwrap().settle(10);
        assert!((90..=91).contains(&remaining(&quota)));

        // An unstarted request that fails is refunded in full; one that
        // started streaming keeps its estimate.
        let (_, refunded) = quota.reserve_tokens(tenant, 50).unwrap();
        drop(refunded);
        assert!((90..=91).contains(&remaining(&quota)));
        let (_, mut kept) = quota.reserve_tokens(tenant, 50).unwrap();
        kept.as_mut().unwrap().mark_started();
        drop(kept);
        assert!((40..=41).contains(&remaining(&quota)));

        // An estimate beyond the balance is refused with the wait for it; an
        // estimate beyond a whole minute's budget needs only a full bucket.
        let (decision, none) = quota.reserve_tokens(tenant, 80).unwrap();
        assert!(none.is_none());
        let TokenDecision::Limited { retry_after } = decision else {
            panic!("an 80-token estimate exceeds a 40-token balance");
        };
        assert!(retry_after > Duration::from_secs(20), "{retry_after:?}");
        let fresh = key(2);
        let (decision, reservation) = quota.reserve_tokens(fresh, 500).unwrap();
        assert_eq!(decision, TokenDecision::Allowed);
        drop(reservation);
    }

    #[test]
    fn without_a_token_limit_nothing_is_reserved() {
        let quota = GatewayRateQuota::new(GatewayRateQuotaConfig::new(600, 64, 8).unwrap());
        let (decision, reservation) = quota.reserve_tokens(key(1), u64::MAX).unwrap();
        assert_eq!(decision, TokenDecision::Allowed);
        assert!(reservation.is_none());
        assert!(quota.snapshot(key(1)).unwrap().tokens.is_none());
    }

    #[test]
    fn peek_reports_the_limit_without_spending() {
        let quota = GatewayRateQuota::new(GatewayRateQuotaConfig::new(1, 1, 8).unwrap());
        for _ in 0..3 {
            assert_eq!(quota.peek(key(1)).unwrap(), GatewayRateDecision::Allowed);
        }
        assert_eq!(quota.check(key(1)).unwrap(), GatewayRateDecision::Allowed);
        assert_eq!(quota.peek(key(1)).unwrap(), GatewayRateDecision::Limited);
        let requests = quota.snapshot(key(1)).unwrap().requests;
        assert_eq!((requests.limit, requests.remaining), (1, 0));
        assert!(requests.retry_after > Duration::from_secs(59));
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

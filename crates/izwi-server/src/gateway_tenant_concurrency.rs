//! Bounded process-local ownership for tenant inference work.
//!
//! A public response ending is not proof that an accepted worker invocation
//! stopped. Bound leases therefore remain in this table until an authenticated
//! terminal event, terminal attempt query, or cancellation response proves
//! teardown. Fleet-wide ownership and gateway-crash recovery belong to the
//! shared Phase 8 authority; this module intentionally does not pretend that a
//! process-local table provides either property.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use izwi_serving_client::{WorkerClient, WorkerClientError};
use izwi_serving_protocol::AttemptIdentity;

const TENANT_MAX_CONCURRENT_ENV: &str = "IZWI_GATEWAY_TENANT_MAX_CONCURRENT";
const MAX_ENV_VALUE_BYTES: usize = 20;
const DEFAULT_TENANT_MAX_CONCURRENT: usize = 8;
pub const MAX_TENANT_CONCURRENT_WORK: usize = 100_000;

#[cfg(not(test))]
const INITIAL_RECONCILE_DELAY: Duration = Duration::from_millis(250);
#[cfg(test)]
const INITIAL_RECONCILE_DELAY: Duration = Duration::from_millis(5);
const MAX_RECONCILE_DELAY: Duration = Duration::from_secs(5);
#[cfg(not(test))]
const MAX_RECONCILE_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(test)]
const MAX_RECONCILE_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayTenantConcurrencyConfig {
    max_active_per_tenant: usize,
    max_owned_work: usize,
}

impl GatewayTenantConcurrencyConfig {
    pub fn max_active_per_tenant(&self) -> usize {
        self.max_active_per_tenant
    }
    pub fn max_owned_work(&self) -> usize {
        self.max_owned_work
    }
    pub fn new(
        max_active_per_tenant: usize,
        max_owned_work: usize,
    ) -> Result<Self, GatewayTenantConcurrencyConfigError> {
        if max_owned_work == 0
            || max_owned_work > MAX_TENANT_CONCURRENT_WORK
            || max_owned_work > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err(GatewayTenantConcurrencyConfigError::OwnedWork);
        }
        if max_active_per_tenant == 0
            || max_active_per_tenant > MAX_TENANT_CONCURRENT_WORK
            || max_active_per_tenant > max_owned_work
        {
            return Err(GatewayTenantConcurrencyConfigError::TenantLimit);
        }
        Ok(Self {
            max_active_per_tenant,
            max_owned_work,
        })
    }

    pub fn from_env(max_owned_work: usize) -> Result<Self, GatewayTenantConcurrencyConfigError> {
        let maximum = match std::env::var_os(TENANT_MAX_CONCURRENT_ENV) {
            None => DEFAULT_TENANT_MAX_CONCURRENT.min(max_owned_work),
            Some(raw) => raw
                .to_str()
                .filter(|value| {
                    !value.is_empty()
                        && value.len() <= MAX_ENV_VALUE_BYTES
                        && value.bytes().all(|byte| byte.is_ascii_digit())
                })
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or(GatewayTenantConcurrencyConfigError::TenantLimit)?,
        };
        Self::new(maximum, max_owned_work)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GatewayTenantConcurrencyConfigError {
    #[error("IZWI_GATEWAY_TENANT_MAX_CONCURRENT must be a bounded non-zero integer no greater than gateway max in flight")]
    TenantLimit,
    #[error("gateway concurrent-work ownership capacity must be a bounded non-zero integer")]
    OwnedWork,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum TenantWorkAdmissionError {
    #[error("tenant concurrent-work limit is exhausted")]
    TenantLimit,
    #[error("gateway concurrent-work ownership capacity is exhausted")]
    OwnershipCapacity,
    #[error("gateway concurrent-work ownership state is unavailable")]
    StateUnavailable,
}

#[derive(Clone)]
pub(crate) struct GatewayTenantConcurrency {
    inner: Arc<TenantConcurrencyInner>,
}

impl fmt::Debug for GatewayTenantConcurrency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayTenantConcurrency")
            .field(
                "max_active_per_tenant",
                &self.inner.config.max_active_per_tenant,
            )
            .field("max_owned_work", &self.inner.config.max_owned_work)
            .field("active_owned_work", &self.active_owned_work())
            .finish()
    }
}

struct TenantConcurrencyInner {
    config: GatewayTenantConcurrencyConfig,
    state: Mutex<TenantConcurrencyState>,
    active_owned_work: AtomicU64,
}

#[derive(Default)]
struct TenantConcurrencyState {
    next_lease_id: u64,
    tenant_counts: BTreeMap<[u8; 32], usize>,
    records: BTreeMap<u64, TenantWorkRecord>,
}

struct TenantWorkRecord {
    tenant_key: [u8; 32],
    attempt: Option<BoundAttempt>,
    reconciliation_started: bool,
}

#[derive(Clone)]
struct BoundAttempt {
    client: WorkerClient,
    identity: AttemptIdentity,
}

impl GatewayTenantConcurrency {
    pub(crate) fn new(config: GatewayTenantConcurrencyConfig) -> Self {
        Self {
            inner: Arc::new(TenantConcurrencyInner {
                config,
                state: Mutex::new(TenantConcurrencyState::default()),
                active_owned_work: AtomicU64::new(0),
            }),
        }
    }

    pub(crate) fn try_reserve(
        &self,
        tenant_key: [u8; 32],
    ) -> Result<UnboundTenantWorkLease, TenantWorkAdmissionError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| TenantWorkAdmissionError::StateUnavailable)?;
        if state.records.len() >= self.inner.config.max_owned_work {
            return Err(TenantWorkAdmissionError::OwnershipCapacity);
        }
        if state.tenant_counts.get(&tenant_key).copied().unwrap_or(0)
            >= self.inner.config.max_active_per_tenant
        {
            return Err(TenantWorkAdmissionError::TenantLimit);
        }
        let lease_id = next_lease_id(&mut state);
        *state.tenant_counts.entry(tenant_key).or_default() += 1;
        state.records.insert(
            lease_id,
            TenantWorkRecord {
                tenant_key,
                attempt: None,
                reconciliation_started: false,
            },
        );
        self.inner.active_owned_work.fetch_add(1, Ordering::Relaxed);
        Ok(UnboundTenantWorkLease {
            core: Some(TenantLeaseCore {
                owner: self.clone(),
                lease_id,
            }),
        })
    }

    pub(crate) fn active_owned_work(&self) -> u64 {
        self.inner.active_owned_work.load(Ordering::Relaxed)
    }

    fn bind(&self, lease_id: u64, client: WorkerClient, identity: AttemptIdentity) {
        let mut state = lock_recover(&self.inner.state);
        if let Some(record) = state.records.get_mut(&lease_id) {
            record.attempt = Some(BoundAttempt { client, identity });
            record.reconciliation_started = false;
        }
    }

    fn unbind(&self, lease_id: u64) {
        let mut state = lock_recover(&self.inner.state);
        if let Some(record) = state.records.get_mut(&lease_id) {
            record.attempt = None;
            record.reconciliation_started = false;
        }
    }

    fn release(&self, lease_id: u64) {
        let mut state = lock_recover(&self.inner.state);
        self.release_locked(&mut state, lease_id);
    }

    fn release_if_attempt(&self, lease_id: u64, identity: &AttemptIdentity) -> bool {
        let mut state = lock_recover(&self.inner.state);
        let matches = state
            .records
            .get(&lease_id)
            .and_then(|record| record.attempt.as_ref())
            .is_some_and(|attempt| attempt.identity == *identity);
        if matches {
            self.release_locked(&mut state, lease_id);
        }
        matches
    }

    fn release_locked(&self, state: &mut TenantConcurrencyState, lease_id: u64) {
        let Some(record) = state.records.remove(&lease_id) else {
            return;
        };
        if let Some(count) = state.tenant_counts.get_mut(&record.tenant_key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.tenant_counts.remove(&record.tenant_key);
            }
        }
        self.inner.active_owned_work.fetch_sub(1, Ordering::Relaxed);
    }

    fn detach_and_reconcile(&self, lease_id: u64) {
        let attempt = {
            let mut state = lock_recover(&self.inner.state);
            let Some(record) = state.records.get_mut(&lease_id) else {
                return;
            };
            if record.reconciliation_started {
                return;
            }
            let Some(attempt) = record.attempt.clone() else {
                self.release_locked(&mut state, lease_id);
                return;
            };
            record.reconciliation_started = true;
            attempt
        };
        let owner = self.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                reconcile_until_stopped(owner, lease_id, attempt).await;
            });
        }
        // Without a runtime the central record deliberately remains retained.
    }

    fn attempt_still_owned(&self, lease_id: u64, identity: &AttemptIdentity) -> bool {
        let state = lock_recover(&self.inner.state);
        state
            .records
            .get(&lease_id)
            .and_then(|record| record.attempt.as_ref())
            .is_some_and(|attempt| attempt.identity == *identity)
    }
}

async fn reconcile_until_stopped(
    owner: GatewayTenantConcurrency,
    lease_id: u64,
    attempt: BoundAttempt,
) {
    let mut delay = INITIAL_RECONCILE_DELAY;
    let start = tokio::time::Instant::now();
    loop {
        if !owner.attempt_still_owned(lease_id, &attempt.identity) {
            return;
        }
        // Exact cancellation is idempotent. Reissue it after transient
        // transport failures instead of assuming one best-effort request will
        // eventually stop the admitted execution.
        match attempt.client.cancel_attempt(&attempt.identity).await {
            Ok(response) if response.disposition.confirms_execution_stopped() => {
                owner.release_if_attempt(lease_id, &attempt.identity);
                return;
            }
            Err(WorkerClientError::HttpStatus { status, .. }) if status.as_u16() == 409 => {
                // A 409 Conflict from cancel_attempt indicates the worker restarted under a
                // new IncarnationId or the attempt's incarnation is dead. Since execution cannot
                // survive across worker incarnation restarts, this confirms execution has stopped.
                owner.release_if_attempt(lease_id, &attempt.identity);
                return;
            }
            _ => {}
        }
        match attempt.client.query_attempt(&attempt.identity).await {
            Ok(response) if response.state.proves_execution_stopped() => {
                owner.release_if_attempt(lease_id, &attempt.identity);
                return;
            }
            Err(WorkerClientError::HttpStatus { status, .. }) if status.as_u16() == 409 => {
                // Incarnation mismatch on query confirms the target incarnation is dead.
                owner.release_if_attempt(lease_id, &attempt.identity);
                return;
            }
            _ => {}
        }
        if start.elapsed() >= MAX_RECONCILE_TIMEOUT {
            tracing::warn!(
                lease_id,
                attempt = ?attempt.identity,
                "reconciliation timed out waiting for worker teardown proof; reclaiming tenant concurrency lease"
            );
            owner.release_if_attempt(lease_id, &attempt.identity);
            return;
        }
        tokio::time::sleep(delay).await;
        delay = delay.saturating_mul(2).min(MAX_RECONCILE_DELAY);
    }
}

fn next_lease_id(state: &mut TenantConcurrencyState) -> u64 {
    loop {
        state.next_lease_id = state.next_lease_id.wrapping_add(1).max(1);
        if !state.records.contains_key(&state.next_lease_id) {
            return state.next_lease_id;
        }
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct TenantLeaseCore {
    owner: GatewayTenantConcurrency,
    lease_id: u64,
}

pub(crate) struct UnboundTenantWorkLease {
    core: Option<TenantLeaseCore>,
}

impl fmt::Debug for UnboundTenantWorkLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnboundTenantWorkLease")
            .field("bound", &false)
            .finish()
    }
}

impl UnboundTenantWorkLease {
    pub(crate) fn bind(
        mut self,
        client: WorkerClient,
        identity: AttemptIdentity,
    ) -> BoundTenantWorkLease {
        let core = self.core.take().expect("tenant work lease is live");
        core.owner.bind(core.lease_id, client, identity);
        BoundTenantWorkLease { core: Some(core) }
    }
}

impl Drop for UnboundTenantWorkLease {
    fn drop(&mut self) {
        if let Some(core) = self.core.take() {
            core.owner.release(core.lease_id);
        }
    }
}

pub(crate) struct BoundTenantWorkLease {
    core: Option<TenantLeaseCore>,
}

impl fmt::Debug for BoundTenantWorkLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundTenantWorkLease")
            .field("bound", &true)
            .finish()
    }
}

impl BoundTenantWorkLease {
    pub(crate) fn prove_unaccepted(mut self) -> UnboundTenantWorkLease {
        let core = self.core.take().expect("tenant work lease is live");
        core.owner.unbind(core.lease_id);
        UnboundTenantWorkLease { core: Some(core) }
    }

    pub(crate) fn confirm_stopped(mut self) {
        if let Some(core) = self.core.take() {
            core.owner.release(core.lease_id);
        }
    }
}

impl Drop for BoundTenantWorkLease {
    fn drop(&mut self) {
        if let Some(core) = self.core.take() {
            core.owner.detach_and_reconcile(core.lease_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(per_tenant: usize, total: usize) -> GatewayTenantConcurrencyConfig {
        GatewayTenantConcurrencyConfig::new(per_tenant, total).unwrap()
    }

    #[test]
    fn admission_is_atomic_per_tenant_and_globally_bounded() {
        let ownership = GatewayTenantConcurrency::new(config(1, 2));
        let first = ownership.try_reserve([1; 32]).unwrap();
        assert_eq!(
            ownership.try_reserve([1; 32]).unwrap_err(),
            TenantWorkAdmissionError::TenantLimit
        );
        let second = ownership.try_reserve([2; 32]).unwrap();
        assert_eq!(
            ownership.try_reserve([3; 32]).unwrap_err(),
            TenantWorkAdmissionError::OwnershipCapacity
        );
        assert_eq!(ownership.active_owned_work(), 2);
        drop(first);
        drop(second);
        assert_eq!(ownership.active_owned_work(), 0);
        assert!(lock_recover(&ownership.inner.state)
            .tenant_counts
            .is_empty());
    }

    #[test]
    fn configuration_is_bounded_and_debug_redacts_identity() {
        assert!(GatewayTenantConcurrencyConfig::new(0, 1).is_err());
        assert!(GatewayTenantConcurrencyConfig::new(2, 1).is_err());
        assert!(GatewayTenantConcurrencyConfig::new(1, 0).is_err());
        assert!(GatewayTenantConcurrencyConfig::new(1, MAX_TENANT_CONCURRENT_WORK + 1).is_err());
        let ownership = GatewayTenantConcurrency::new(config(1, 1));
        let lease = ownership.try_reserve([0xabu8; 32]).unwrap();
        let debug = format!("{ownership:?} {lease:?}");
        assert!(!debug.contains("abababab"));
    }

    #[test]
    fn environment_limit_is_bounded_by_global_ownership() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(TENANT_MAX_CONCURRENT_ENV);
        assert_eq!(
            GatewayTenantConcurrencyConfig::from_env(4)
                .unwrap()
                .max_active_per_tenant,
            4
        );

        std::env::set_var(TENANT_MAX_CONCURRENT_ENV, "2");
        assert_eq!(
            GatewayTenantConcurrencyConfig::from_env(4)
                .unwrap()
                .max_active_per_tenant,
            2
        );
        for invalid in ["0", "5", "not-a-number"] {
            std::env::set_var(TENANT_MAX_CONCURRENT_ENV, invalid);
            assert!(GatewayTenantConcurrencyConfig::from_env(4).is_err());
        }
        std::env::remove_var(TENANT_MAX_CONCURRENT_ENV);
    }

    #[tokio::test]
    async fn reconcile_releases_lease_on_incarnation_conflict() {
        use axum::http::StatusCode;
        use axum::routing::post;
        use axum::Router;
        use izwi_serving_client::WorkerClientConfig;
        use izwi_serving_protocol::{
            AttemptId, AttemptIdentity, CallerId, CredentialId, DeploymentId, IncarnationId,
            ModelGeneration, RequestId, ServiceBearerToken, ServiceCredentials, TenantId,
        };

        let app = Router::new().route(
            "/internal/v1/invocations/{attempt_id}/cancel",
            post(|| async { StatusCode::CONFLICT }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let credentials = ServiceCredentials {
            credential_id: CredentialId::new("test").unwrap(),
            bearer_token: ServiceBearerToken::new("test-token").unwrap(),
        };
        let client = WorkerClient::new(
            &format!("http://{addr}"),
            credentials,
            WorkerClientConfig::default(),
        )
        .unwrap();

        let identity = AttemptIdentity {
            request_id: RequestId::new("req-conflict").unwrap(),
            attempt_id: AttemptId::new("att-conflict").unwrap(),
            tenant_id: TenantId::new("tenant-1").unwrap(),
            caller_id: CallerId::new("caller-1").unwrap(),
            incarnation_id: IncarnationId::new("inc-conflict").unwrap(),
            deployment_id: DeploymentId::new("dep-conflict").unwrap(),
            model_generation: ModelGeneration::new(1).unwrap(),
        };

        let ownership = GatewayTenantConcurrency::new(config(1, 1));
        let lease = ownership.try_reserve([1; 32]).unwrap();
        let bound = lease.bind(client, identity);
        assert_eq!(ownership.active_owned_work(), 1);

        drop(bound);

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(ownership.active_owned_work(), 0);
        assert!(lock_recover(&ownership.inner.state)
            .tenant_counts
            .is_empty());
    }

    #[tokio::test]
    async fn reconcile_releases_lease_on_orphan_timeout() {
        use izwi_serving_client::WorkerClientConfig;
        use izwi_serving_protocol::{
            AttemptId, AttemptIdentity, CallerId, CredentialId, DeploymentId, IncarnationId,
            ModelGeneration, RequestId, ServiceBearerToken, ServiceCredentials, TenantId,
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let credentials = ServiceCredentials {
            credential_id: CredentialId::new("test").unwrap(),
            bearer_token: ServiceBearerToken::new("test-token").unwrap(),
        };
        let client_config = WorkerClientConfig {
            connect_timeout: Duration::from_millis(20),
            request_timeout: Duration::from_millis(20),
            ..WorkerClientConfig::default()
        };
        let client =
            WorkerClient::new(&format!("http://{addr}"), credentials, client_config).unwrap();

        let identity = AttemptIdentity {
            request_id: RequestId::new("req-orphan").unwrap(),
            attempt_id: AttemptId::new("att-orphan").unwrap(),
            tenant_id: TenantId::new("tenant-1").unwrap(),
            caller_id: CallerId::new("caller-1").unwrap(),
            incarnation_id: IncarnationId::new("inc-orphan").unwrap(),
            deployment_id: DeploymentId::new("dep-orphan").unwrap(),
            model_generation: ModelGeneration::new(1).unwrap(),
        };

        let ownership = GatewayTenantConcurrency::new(config(1, 1));
        let lease = ownership.try_reserve([2; 32]).unwrap();
        let bound = lease.bind(client, identity);
        assert_eq!(ownership.active_owned_work(), 1);

        drop(bound);

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(ownership.active_owned_work(), 0);
        assert!(lock_recover(&ownership.inner.state)
            .tenant_counts
            .is_empty());
    }
}

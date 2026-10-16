//! Bounded restarts and quarantine for one supervised worker.

use crate::{LifecycleError, RestartPolicy, ValidatedNodeConfig};
use izwi_serving_protocol::WorkerId;
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartDecision {
    RestartAfter(Duration),
    Quarantine,
}

/// Bounded crash history and deterministic per-worker jitter.
#[derive(Debug, Clone)]
pub struct RestartController {
    policy: RestartPolicy,
    failures: VecDeque<Duration>,
    failure_streak: u32,
    jitter_seed: u64,
    jitter_sequence: u64,
    quarantined: bool,
}

impl RestartController {
    pub fn for_worker(
        node: &ValidatedNodeConfig,
        worker_id: &WorkerId,
    ) -> Result<Self, LifecycleError> {
        if node.worker(worker_id).is_none() {
            return Err(LifecycleError::UnknownWorker(worker_id.clone()));
        }
        let digest = Sha256::digest(worker_id.as_str().as_bytes());
        let jitter_seed = u64::from_le_bytes(digest[..8].try_into().expect("fixed digest width"));
        Ok(Self {
            policy: node.config().restart.clone(),
            failures: VecDeque::with_capacity(
                usize::from(node.config().restart.max_restarts_per_window).min(100),
            ),
            failure_streak: 0,
            jitter_seed,
            jitter_sequence: 0,
            quarantined: false,
        })
    }

    #[cfg(test)]
    fn with_seed(policy: RestartPolicy, jitter_seed: u64) -> Self {
        Self {
            failures: VecDeque::with_capacity(usize::from(policy.max_restarts_per_window)),
            policy,
            failure_streak: 0,
            jitter_seed,
            jitter_sequence: 0,
            quarantined: false,
        }
    }

    pub fn record_failure(&mut self, now: Duration, worker_uptime: Duration) -> RestartDecision {
        if self.quarantined {
            return RestartDecision::Quarantine;
        }
        if worker_uptime >= Duration::from_millis(self.policy.stable_reset_ms) {
            self.failures.clear();
            self.failure_streak = 0;
        }
        let window = Duration::from_millis(self.policy.restart_window_ms);
        while self
            .failures
            .front()
            .is_some_and(|recorded| now.saturating_sub(*recorded) >= window)
        {
            self.failures.pop_front();
        }
        if self.failures.len() >= usize::from(self.policy.max_restarts_per_window) {
            self.quarantined = true;
            return RestartDecision::Quarantine;
        }
        self.failures.push_back(now);

        let exponent = self.failure_streak.min(63);
        let multiplier = 1_u64.checked_shl(exponent).unwrap_or(u64::MAX);
        let base = self
            .policy
            .initial_backoff_ms
            .saturating_mul(multiplier)
            .min(self.policy.maximum_backoff_ms);
        self.failure_streak = self.failure_streak.saturating_add(1);
        let delay = jittered_delay_ms(
            base,
            self.policy.maximum_backoff_ms,
            self.policy.jitter_percent,
            splitmix64(self.jitter_seed ^ self.jitter_sequence),
        );
        self.jitter_sequence = self.jitter_sequence.wrapping_add(1);
        RestartDecision::RestartAfter(Duration::from_millis(delay))
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn recent_failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Quarantine is fail-closed and can only be cleared by an explicit operator action.
    pub fn clear_quarantine(&mut self) {
        self.failures.clear();
        self.failure_streak = 0;
        self.quarantined = false;
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn jittered_delay_ms(base: u64, maximum: u64, percent: u8, sample: u64) -> u64 {
    let span = u64::try_from((u128::from(base) * u128::from(percent)) / 100).unwrap_or(u64::MAX);
    if span == 0 {
        return base.min(maximum);
    }
    let width = span.saturating_mul(2).saturating_add(1);
    let offset = i128::from(sample % width) - i128::from(span);
    let adjusted = (i128::from(base) + offset).max(1);
    u64::try_from(adjusted).unwrap_or(maximum).min(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn restart_policy() -> RestartPolicy {
        RestartPolicy {
            initial_backoff_ms: 100,
            maximum_backoff_ms: 800,
            restart_window_ms: 1_000,
            stable_reset_ms: 5_000,
            max_restarts_per_window: 3,
            jitter_percent: 0,
        }
    }

    #[test]
    fn restart_backoff_is_bounded_and_quarantines_at_budget() {
        let mut controller = RestartController::with_seed(restart_policy(), 7);
        assert_eq!(
            controller.record_failure(Duration::ZERO, Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(100))
        );
        assert_eq!(
            controller.record_failure(Duration::from_millis(10), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(200))
        );
        assert_eq!(
            controller.record_failure(Duration::from_millis(20), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(400))
        );
        assert_eq!(
            controller.record_failure(Duration::from_millis(30), Duration::ZERO),
            RestartDecision::Quarantine
        );
        assert!(controller.is_quarantined());
        assert_eq!(controller.recent_failure_count(), 3);
    }

    #[test]
    fn stable_uptime_resets_backoff_and_expired_failures_leave_window() {
        let mut controller = RestartController::with_seed(restart_policy(), 7);
        let _ = controller.record_failure(Duration::ZERO, Duration::ZERO);
        let _ = controller.record_failure(Duration::from_millis(10), Duration::ZERO);
        assert_eq!(
            controller.record_failure(Duration::from_millis(2_000), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(400))
        );
        assert_eq!(controller.recent_failure_count(), 1);
        assert_eq!(
            controller.record_failure(Duration::from_millis(2_100), Duration::from_secs(5)),
            RestartDecision::RestartAfter(Duration::from_millis(100))
        );
    }

    #[test]
    fn deterministic_jitter_never_exceeds_maximum() {
        let mut policy = restart_policy();
        policy.jitter_percent = 25;
        let mut first = RestartController::with_seed(policy.clone(), 42);
        let mut second = RestartController::with_seed(policy, 42);
        for index in 0..3 {
            let now = Duration::from_millis(index * 10);
            let left = first.record_failure(now, Duration::ZERO);
            let right = second.record_failure(now, Duration::ZERO);
            assert_eq!(left, right);
            let RestartDecision::RestartAfter(delay) = left else {
                panic!("restart budget unexpectedly exhausted")
            };
            assert!(delay <= Duration::from_millis(800));
            assert!(!delay.is_zero());
        }
    }
}

//! Bounded restarts and quarantine for one supervised worker.
//!
//! A worker that exhausts its restart budget is quarantined. Quarantine is
//! not permanent: the supervisor retries it once every
//! `restart.quarantine_retry_ms` (like Kubernetes' CrashLoopBackOff), and a
//! retry only ends the quarantine once the worker stays up for
//! `stable_reset_ms`. An operator can clear it at once with SIGHUP (every
//! worker) or `--clear-quarantine` (one worker or all), which leaves a
//! request file in the runtime directory for the running supervisor.

use crate::{LifecycleError, RestartPolicy, ValidatedNodeConfig};
use izwi_serving_protocol::WorkerId;
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, path::Path, time::Duration};

/// Directory in the runtime directory that holds quarantine clear requests.
pub const QUARANTINE_CLEAR_DIRECTORY: &str = "quarantine-clear";
const CLEAR_ALL_REQUEST: &str = "all.request";
const WORKER_REQUEST_SUFFIX: &str = ".worker";
const MAX_CLEAR_REQUESTS: usize = 256;

/// Which quarantines an operator asked to clear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuarantineClear {
    All,
    Worker(WorkerId),
}

/// Leaves a clear request for the running supervisor, which consumes it
/// within about a second.
pub fn request_quarantine_clear(
    runtime_directory: &Path,
    target: &QuarantineClear,
) -> std::io::Result<()> {
    let directory = runtime_directory.join(QUARANTINE_CLEAR_DIRECTORY);
    std::fs::create_dir_all(&directory)?;
    let name = match target {
        QuarantineClear::All => CLEAR_ALL_REQUEST.to_string(),
        QuarantineClear::Worker(worker) => format!("{worker}{WORKER_REQUEST_SUFFIX}"),
    };
    crate::approvals_file::replace_file(&directory.join(name), b"clear\n")
}

/// Takes the pending clear requests, removing their files. Unrecognized
/// files are removed and ignored.
pub fn take_quarantine_clear_requests(runtime_directory: &Path) -> Vec<QuarantineClear> {
    let directory = runtime_directory.join(QUARANTINE_CLEAR_DIRECTORY);
    let Ok(entries) = std::fs::read_dir(&directory) else {
        return Vec::new();
    };
    let mut requests = Vec::new();
    for entry in entries.flatten().take(MAX_CLEAR_REQUESTS) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            // A request still being written.
            continue;
        }
        let _ = std::fs::remove_file(entry.path());
        if name == CLEAR_ALL_REQUEST {
            requests.push(QuarantineClear::All);
        } else if let Some(worker) = name
            .strip_suffix(WORKER_REQUEST_SUFFIX)
            .and_then(|worker| WorkerId::new(worker).ok())
        {
            requests.push(QuarantineClear::Worker(worker));
        }
    }
    requests
}

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
    /// Released from quarantine by a retry that has not yet stayed up for
    /// `stable_reset_ms`: an early failure quarantines it again.
    probation: bool,
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
            probation: false,
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
            probation: false,
        }
    }

    pub fn record_failure(&mut self, now: Duration, worker_uptime: Duration) -> RestartDecision {
        if worker_uptime >= Duration::from_millis(self.policy.stable_reset_ms) {
            self.failures.clear();
            self.failure_streak = 0;
            self.probation = false;
        }
        if self.quarantined || self.probation {
            self.quarantined = true;
            self.probation = false;
            return RestartDecision::Quarantine;
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

    /// How long a quarantined worker waits for its next retry; `None` when
    /// only an operator can clear it.
    pub fn quarantine_retry(&self) -> Option<Duration> {
        (self.policy.quarantine_retry_ms > 0)
            .then(|| Duration::from_millis(self.policy.quarantine_retry_ms))
    }

    /// A launch reached readiness. A quarantine retry that did leaves
    /// quarantine on probation.
    pub fn note_ready(&mut self) {
        if self.quarantined {
            self.quarantined = false;
            self.probation = true;
        }
    }

    pub fn recent_failure_count(&self) -> usize {
        self.failures.len()
    }

    /// An operator cleared the quarantine: the worker gets a fresh budget.
    pub fn clear_quarantine(&mut self) {
        self.failures.clear();
        self.failure_streak = 0;
        self.probation = false;
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
            quarantine_retry_ms: 60_000,
        }
    }

    #[test]
    fn a_quarantine_retry_ends_quarantine_only_after_stable_uptime() {
        let mut controller = RestartController::with_seed(restart_policy(), 7);
        for now in [0, 10, 20] {
            let _ = controller.record_failure(Duration::from_millis(now), Duration::ZERO);
        }
        assert_eq!(
            controller.record_failure(Duration::from_millis(30), Duration::ZERO),
            RestartDecision::Quarantine
        );
        assert_eq!(controller.quarantine_retry(), Some(Duration::from_secs(60)));

        // A retry that fails before readiness stays quarantined.
        assert_eq!(
            controller.record_failure(Duration::from_secs(60), Duration::from_millis(5)),
            RestartDecision::Quarantine
        );
        // A retry that reaches readiness is on probation: an early crash
        // quarantines it again...
        controller.note_ready();
        assert!(!controller.is_quarantined());
        assert_eq!(
            controller.record_failure(Duration::from_secs(120), Duration::from_secs(1)),
            RestartDecision::Quarantine
        );
        // ...and one that stayed up for stable_reset_ms is back to normal.
        controller.note_ready();
        assert_eq!(
            controller.record_failure(Duration::from_secs(200), Duration::from_secs(5)),
            RestartDecision::RestartAfter(Duration::from_millis(100))
        );
    }

    #[test]
    fn an_operator_clear_restores_the_full_budget() {
        let mut controller = RestartController::with_seed(restart_policy(), 7);
        for now in [0, 10, 20, 30] {
            let _ = controller.record_failure(Duration::from_millis(now), Duration::ZERO);
        }
        assert!(controller.is_quarantined());
        controller.clear_quarantine();
        assert!(!controller.is_quarantined());
        assert_eq!(controller.recent_failure_count(), 0);
        assert_eq!(
            controller.record_failure(Duration::from_millis(40), Duration::ZERO),
            RestartDecision::RestartAfter(Duration::from_millis(100))
        );
    }

    #[test]
    fn clear_requests_round_trip_through_the_runtime_directory() {
        let runtime = tempfile::tempdir().unwrap();
        assert!(take_quarantine_clear_requests(runtime.path()).is_empty());
        let worker = WorkerId::new("worker-a").unwrap();
        request_quarantine_clear(runtime.path(), &QuarantineClear::Worker(worker.clone())).unwrap();
        request_quarantine_clear(runtime.path(), &QuarantineClear::All).unwrap();
        std::fs::write(
            runtime
                .path()
                .join(QUARANTINE_CLEAR_DIRECTORY)
                .join("stray"),
            b"x",
        )
        .unwrap();
        let mut requests = take_quarantine_clear_requests(runtime.path());
        requests.sort_by_key(|request| format!("{request:?}"));
        assert_eq!(
            requests,
            vec![QuarantineClear::All, QuarantineClear::Worker(worker)]
        );
        assert!(
            take_quarantine_clear_requests(runtime.path()).is_empty(),
            "requests are consumed once"
        );
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

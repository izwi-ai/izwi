//! Liveness probing of a ready worker.
//!
//! Readiness proves a worker came up; nothing afterwards proved it still
//! works. A worker whose device or runtime wedged keeps its process and its
//! listener, so neither the exit watch nor the gateway's stale-status
//! filter brings its capacity back. The probe asks the worker's status
//! endpoint on a period and judges each answer; a run of failures restarts
//! the worker, like a Kubernetes liveness probe.
//!
//! Only process-level failure counts: no answer within the timeout, an
//! answer from another identity, or a runtime the worker itself reports as
//! failed. A deployment that is merely not ready (a drain-recoverable fault)
//! is a readiness matter the gateway already routes around.

use izwi_serving_client::WorkerClientError;
use izwi_serving_protocol::{WorkerProcessState, WorkerStatus};

use crate::{ExpectedWorkerIdentity, LivenessPolicy};

/// One judged probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeVerdict {
    Alive,
    Failed(String),
}

/// Judges one status probe of the expected incarnation. `None` is a probe
/// that did not answer within its timeout.
pub fn judge(
    expected: &ExpectedWorkerIdentity,
    answer: Option<Result<WorkerStatus, WorkerClientError>>,
) -> ProbeVerdict {
    let status = match answer {
        None => return ProbeVerdict::Failed("status did not answer in time".to_string()),
        Some(Err(error)) => return ProbeVerdict::Failed(format!("status failed: {error}")),
        Some(Ok(status)) => status,
    };
    if &status.worker_id != expected.worker_id()
        || &status.node_id != expected.node_id()
        || &status.incarnation_id != expected.incarnation_id()
    {
        return ProbeVerdict::Failed(
            "status came from another worker identity or incarnation".to_string(),
        );
    }
    if status.process_state == WorkerProcessState::Failed {
        return ProbeVerdict::Failed("the worker reports a failed runtime".to_string());
    }
    ProbeVerdict::Alive
}

/// What a run of probes means for the worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Liveness {
    Alive,
    /// Failing, but below the threshold.
    Suspect {
        failures: u32,
        reason: String,
    },
    /// The threshold was reached: restart the worker.
    Dead {
        reason: String,
    },
}

/// Counts consecutive probe failures for one incarnation.
#[derive(Debug, Clone)]
pub struct LivenessTracker {
    threshold: u32,
    failures: u32,
}

impl LivenessTracker {
    pub fn new(policy: &LivenessPolicy) -> Self {
        Self {
            threshold: policy.failure_threshold,
            failures: 0,
        }
    }

    pub fn observe(&mut self, verdict: ProbeVerdict) -> Liveness {
        match verdict {
            ProbeVerdict::Alive => {
                self.failures = 0;
                Liveness::Alive
            }
            ProbeVerdict::Failed(reason) => {
                self.failures = self.failures.saturating_add(1);
                if self.failures >= self.threshold {
                    Liveness::Dead { reason }
                } else {
                    Liveness::Suspect {
                        failures: self.failures,
                        reason,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_of_failures_reaches_the_threshold_and_success_resets_it() {
        let mut tracker = LivenessTracker::new(&LivenessPolicy {
            period_ms: 1_000,
            timeout_ms: 500,
            failure_threshold: 3,
        });
        let failed = || ProbeVerdict::Failed("status did not answer in time".to_string());
        assert!(matches!(
            tracker.observe(failed()),
            Liveness::Suspect { failures: 1, .. }
        ));
        assert!(matches!(
            tracker.observe(failed()),
            Liveness::Suspect { failures: 2, .. }
        ));
        assert_eq!(tracker.observe(ProbeVerdict::Alive), Liveness::Alive);
        assert!(matches!(
            tracker.observe(failed()),
            Liveness::Suspect { .. }
        ));
        assert!(matches!(
            tracker.observe(failed()),
            Liveness::Suspect { .. }
        ));
        assert!(matches!(tracker.observe(failed()), Liveness::Dead { .. }));
    }
}

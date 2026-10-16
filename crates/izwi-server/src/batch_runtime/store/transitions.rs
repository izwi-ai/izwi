//! The job and stage state machines: every status change the store makes,
//! and the status sets its SQL predicates are written against.
//!
//! Jobs are created `queued`; stages are created `queued`. The tests at the
//! bottom check the store's SQL for statuses outside these tables, so a
//! predicate cannot quietly name a state nothing writes.

use super::*;

use RuntimeJobStatus as Job;
use RuntimeStageStatus as Stage;

/// Every allowed job status change.
pub(crate) const JOB_TRANSITIONS: &[(RuntimeJobStatus, RuntimeJobStatus)] = &[
    // A worker claims one of the job's stages.
    (Job::Queued, Job::Running),
    (Job::Running, Job::Running),
    (Job::Retrying, Job::Running),
    // A stage failed and waits for its next attempt.
    (Job::Queued, Job::Retrying),
    (Job::Running, Job::Retrying),
    (Job::Retrying, Job::Retrying),
    // Every stage completed.
    (Job::Queued, Job::Completed),
    (Job::Running, Job::Completed),
    (Job::Retrying, Job::Completed),
    // A stage ran out of attempts, or reconciliation failed an orphan.
    (Job::Queued, Job::Failed),
    (Job::Running, Job::Failed),
    (Job::Retrying, Job::Failed),
    // Cancellation, once no attempt is still executing.
    (Job::Queued, Job::Cancelled),
    (Job::Running, Job::Cancelled),
    (Job::Retrying, Job::Cancelled),
    // Manual retry (`retry_job`).
    (Job::Failed, Job::Queued),
    (Job::Cancelled, Job::Queued),
];

/// Every allowed stage status change.
pub(crate) const STAGE_TRANSITIONS: &[(RuntimeStageStatus, RuntimeStageStatus)] = &[
    // A worker claims the stage.
    (Stage::Queued, Stage::Running),
    (Stage::Retrying, Stage::Running),
    // The attempt finishes, yields, or fails with or without retry.
    (Stage::Running, Stage::Completed),
    (Stage::Running, Stage::Queued),
    (Stage::Running, Stage::Retrying),
    (Stage::Running, Stage::Failed),
    // Cancellation, or reconciliation under a terminal parent job.
    (Stage::Queued, Stage::Cancelled),
    (Stage::Running, Stage::Cancelled),
    (Stage::Retrying, Stage::Cancelled),
    // Manual retry (`retry_job`).
    (Stage::Failed, Stage::Retrying),
    (Stage::Cancelled, Stage::Retrying),
];

/// A job no worker will touch again: its only way out is a manual retry.
pub(crate) fn is_terminal_job_status(status: RuntimeJobStatus) -> bool {
    JOB_TRANSITIONS
        .iter()
        .filter(|(from, _)| *from == status)
        .all(|(_, to)| *to == Job::Queued)
}

/// A job whose stages a worker may claim.
pub(crate) fn is_claimable_job_status(status: RuntimeJobStatus) -> bool {
    JOB_TRANSITIONS.contains(&(status, Job::Running))
}

/// A stage with an executing attempt, the only status an attempt can finish,
/// fail or yield from.
pub(crate) fn is_live_stage_status(status: RuntimeStageStatus) -> bool {
    STAGE_TRANSITIONS.contains(&(status, Stage::Completed))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_JOB_STATUSES: [RuntimeJobStatus; 6] = [
        Job::Queued,
        Job::Running,
        Job::Retrying,
        Job::Completed,
        Job::Failed,
        Job::Cancelled,
    ];
    const ALL_STAGE_STATUSES: [RuntimeStageStatus; 6] = [
        Stage::Queued,
        Stage::Running,
        Stage::Retrying,
        Stage::Completed,
        Stage::Failed,
        Stage::Cancelled,
    ];

    #[test]
    fn status_sets_follow_from_the_transition_tables() {
        let terminal = ALL_JOB_STATUSES
            .into_iter()
            .filter(|status| is_terminal_job_status(*status))
            .collect::<Vec<_>>();
        assert_eq!(terminal, [Job::Completed, Job::Failed, Job::Cancelled]);
        let claimable = ALL_JOB_STATUSES
            .into_iter()
            .filter(|status| is_claimable_job_status(*status))
            .collect::<Vec<_>>();
        assert_eq!(claimable, [Job::Queued, Job::Running, Job::Retrying]);
        let live = ALL_STAGE_STATUSES
            .into_iter()
            .filter(|status| is_live_stage_status(*status))
            .collect::<Vec<_>>();
        assert_eq!(live, [Stage::Running]);
        assert!(
            !JOB_TRANSITIONS.iter().any(|(from, _)| *from == Job::Completed),
            "a completed job never changes again"
        );
        for status in ALL_STAGE_STATUSES {
            assert!(
                STAGE_TRANSITIONS
                    .iter()
                    .any(|(from, to)| *from == status || *to == status),
                "{status:?} takes part in a stage transition"
            );
        }
    }

    /// Every status literal the store's SQL compares a job or stage status
    /// against is one of the states above.
    #[test]
    fn store_sql_names_only_live_statuses() {
        let live = ALL_JOB_STATUSES
            .iter()
            .map(|status| status.as_db_value())
            .chain(ALL_STAGE_STATUSES.iter().map(|status| status.as_db_value()))
            .collect::<std::collections::BTreeSet<_>>();
        let sources = [
            ("acceptance.rs", include_str!("acceptance.rs")),
            ("artifacts.rs", include_str!("artifacts.rs")),
            ("idempotency.rs", include_str!("idempotency.rs")),
            ("jobs.rs", include_str!("jobs.rs")),
            ("media.rs", include_str!("media.rs")),
            ("provider_writes.rs", include_str!("provider_writes.rs")),
            ("queue.rs", include_str!("queue.rs")),
            ("reconcile.rs", include_str!("reconcile.rs")),
            ("retention.rs", include_str!("retention.rs")),
            ("speech.rs", include_str!("speech.rs")),
            ("workers.rs", include_str!("workers.rs")),
            (
                "speech_history_store.rs",
                include_str!("../../speech_history_store.rs"),
            ),
            (
                "transcription_store.rs",
                include_str!("../../transcription_store.rs"),
            ),
        ];
        let mut checked = 0;
        for (file, source) in sources {
            for literal in status_literals(source) {
                assert!(
                    live.contains(literal.as_str()),
                    "{file} compares a status with '{literal}', which no transition reaches"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "the scan found the store's predicates ({checked})");
    }

    /// The quoted literals in `status IN (...)`, `status NOT IN (...)`,
    /// `status = '...'` and `status <> '...'`, where `status` is a whole
    /// column name (so `processing_status` and `scan_status` are skipped).
    fn status_literals(source: &str) -> Vec<String> {
        let bytes = source.as_bytes();
        let mut literals = Vec::new();
        let mut from = 0;
        while let Some(offset) = source[from..].find("status") {
            let start = from + offset;
            from = start + "status".len();
            let preceded_by_word = start > 0
                && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
            let followed_by_word = bytes
                .get(from)
                .is_some_and(|next| next.is_ascii_alphanumeric() || *next == b'_');
            if preceded_by_word || followed_by_word {
                continue;
            }
            let rest = source[from..].trim_start();
            let list = if let Some(rest) = rest
                .strip_prefix("NOT IN (")
                .or_else(|| rest.strip_prefix("IN ("))
            {
                rest.split(')').next().unwrap_or_default()
            } else if let Some(rest) = rest.strip_prefix("<>").or_else(|| rest.strip_prefix('=')) {
                let rest = rest.trim_start();
                if !rest.starts_with('\'') {
                    continue;
                }
                rest.split_once(['\n', ' ', ')', ','])
                    .map_or(rest, |(value, _)| value)
            } else {
                continue;
            };
            literals.extend(
                list.split(',')
                    .map(str::trim)
                    .filter(|value| value.starts_with('\''))
                    .map(|value| value.trim_matches('\'').to_string()),
            );
        }
        literals
    }
}

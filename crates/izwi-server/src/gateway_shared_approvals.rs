//! Shared worker-approval file with TTL-cached fresh views for multi-gateway fleets.
//!
//! A single gateway can take its worker approvals from `--gateway-worker-approval`
//! CLI values alone. When operators run multiple gateways against the same
//! worker fleet, those gateways need an authoritative shared configuration so
//! they all approve the same worker set. This module loads bounded worker
//! approvals from one shared UTF-8 file, caches them locally, and refreshes
//! them on a bounded TTL driven by the file's modification time.
//!
//! File format (one approval per line, `#` starts a comment, blank lines are
//! skipped):
//!
//! ```text
//! # pinned loopback replica
//! http://127.0.0.1:9470|chat|LFM2.5-1.2B-Instruct-GGUF|lfm25-cpu-v1|1
//! # versioned fleet approval
//! v1|https://worker-b.internal:9470|node-1|worker-b|chat|LFM2.5-1.2B-Instruct-GGUF|lfm25-cuda-v1|2
//! ```
//!
//! The loader enforces the same bounded per-approval rules as
//! `GatewayWorkerApproval::from_str` plus a hard file-size ceiling and a
//! hard entry ceiling. Refresh is fail-closed: if the file becomes
//! unreadable, oversized, or contains any invalid entry, the previously
//! cached view is retained and the error is surfaced without admission going
//! down.

use std::str::FromStr;

use super::gateway_deployments::GatewayWorkerApproval;

const SHARED_APPROVALS_PATH_ENV: &str = "IZWI_GATEWAY_SHARED_APPROVALS_PATH";
const SHARED_APPROVALS_TTL_MS_ENV: &str = "IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS";
const MAX_SHARED_APPROVALS_FILE_BYTES: u64 = 64 * 1024;
const MAX_SHARED_APPROVAL_ENTRIES: usize = 256;
const DEFAULT_SHARED_APPROVALS_TTL: std::time::Duration = std::time::Duration::from_secs(30);
const MIN_SHARED_APPROVALS_TTL: std::time::Duration = std::time::Duration::from_secs(1);
const MAX_SHARED_APPROVALS_TTL: std::time::Duration = std::time::Duration::from_secs(3600);

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SharedApprovalsError {
    #[error("shared approvals path exceeds its encoded size limit")]
    PathTooLong,
    #[error("shared approvals file cannot be read")]
    Unreadable,
    #[error("shared approvals file exceeds its 64 KiB size limit")]
    FileTooLarge,
    #[error("shared approvals file contains more than 256 entries")]
    TooManyEntries,
    #[error("shared approvals file line {line} is invalid: {detail}")]
    InvalidEntry { line: usize, detail: String },
    #[error("shared approvals TTL must be between 1 and 3600 seconds")]
    InvalidTtl,
}

#[derive(Debug, Clone)]
pub struct SharedApprovalsConfig {
    path: std::path::PathBuf,
    ttl: std::time::Duration,
}

impl SharedApprovalsConfig {
    pub fn from_env() -> Result<Option<Self>, SharedApprovalsError> {
        let Some(raw) = std::env::var_os(SHARED_APPROVALS_PATH_ENV) else {
            return Ok(None);
        };
        if raw.is_empty() || raw.len() > 4096 {
            return Err(SharedApprovalsError::PathTooLong);
        }
        let path = std::path::PathBuf::from(raw);
        if !path.is_absolute() {
            return Err(SharedApprovalsError::PathTooLong);
        }
        let ttl = match std::env::var(SHARED_APPROVALS_TTL_MS_ENV) {
            Ok(raw) => {
                let ms: u64 = raw.parse().map_err(|_| SharedApprovalsError::InvalidTtl)?;
                let ttl = std::time::Duration::from_millis(ms);
                if ttl < MIN_SHARED_APPROVALS_TTL || ttl > MAX_SHARED_APPROVALS_TTL {
                    return Err(SharedApprovalsError::InvalidTtl);
                }
                ttl
            }
            Err(_) => DEFAULT_SHARED_APPROVALS_TTL,
        };
        Ok(Some(Self { path, ttl }))
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn ttl(&self) -> std::time::Duration {
        self.ttl
    }
}

/// A locally cached fresh view of the shared approvals file.
///
/// Refresh is driven by file modification time: the file is re-read only
/// after the TTL elapses. The cached view is retained on refresh failure so
/// a transient file problem cannot take down healthy admission.
#[derive(Debug)]
pub struct SharedApprovalsView {
    config: SharedApprovalsConfig,
    approvals: Vec<GatewayWorkerApproval>,
    last_refresh: Option<(std::time::SystemTime, std::time::Instant)>,
}

impl SharedApprovalsView {
    pub fn load(config: SharedApprovalsConfig) -> Result<Self, SharedApprovalsError> {
        let approvals = read_approvals_file(config.path())?;
        Ok(Self {
            config,
            approvals,
            last_refresh: Some((std::time::SystemTime::now(), std::time::Instant::now())),
        })
    }

    pub fn approvals(&self) -> &[GatewayWorkerApproval] {
        &self.approvals
    }

    pub fn ttl(&self) -> std::time::Duration {
        self.config.ttl()
    }

    /// Refresh the cached view when the TTL has elapsed. On success the
    /// approvals are replaced; on failure the previous view is retained and
    /// the error is returned so the caller can log it without dropping
    /// healthy admission.
    pub fn refresh_if_due(&mut self) -> Result<bool, SharedApprovalsError> {
        let due = match self.last_refresh {
            Some((_, instant)) => instant.elapsed() >= self.config.ttl(),
            None => true,
        };
        if !due {
            return Ok(false);
        }
        let approvals = read_approvals_file(self.config.path())?;
        self.approvals = approvals;
        self.last_refresh = Some((std::time::SystemTime::now(), std::time::Instant::now()));
        Ok(true)
    }
}

fn read_approvals_file(
    path: &std::path::Path,
) -> Result<Vec<GatewayWorkerApproval>, SharedApprovalsError> {
    let metadata = std::fs::metadata(path).map_err(|_| SharedApprovalsError::Unreadable)?;
    if !metadata.is_file() {
        return Err(SharedApprovalsError::Unreadable);
    }
    if metadata.len() > MAX_SHARED_APPROVALS_FILE_BYTES {
        return Err(SharedApprovalsError::FileTooLarge);
    }
    let text = std::fs::read_to_string(path).map_err(|_| SharedApprovalsError::Unreadable)?;
    let mut approvals = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let entry = line.split('#').next().unwrap_or("").trim();
        if entry.is_empty() {
            continue;
        }
        if approvals.len() >= MAX_SHARED_APPROVAL_ENTRIES {
            return Err(SharedApprovalsError::TooManyEntries);
        }
        match GatewayWorkerApproval::from_str(entry) {
            Ok(approval) => approvals.push(approval),
            Err(error) => {
                return Err(SharedApprovalsError::InvalidEntry {
                    line: index + 1,
                    detail: error.to_string(),
                })
            }
        }
    }
    Ok(approvals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("approvals.txt");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.sync_all().unwrap();
        (dir, path)
    }

    #[test]
    fn parses_comments_blank_lines_and_both_approval_forms() {
        let (_dir, path) = write_temp(
            "# loopback replica\nhttp://127.0.0.1:9470|chat|tiny-model|deploy-a|1\n\n# v1 fleet\nv1|https://worker-b.internal:9470|node-1|worker-b|chat|tiny-model|deploy-a|2\n",
        );
        let entries = read_approvals_file(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].deployment_id.as_str(), "deploy-a");
        assert_eq!(entries[1].deployment_id.as_str(), "deploy-a");
    }

    #[test]
    fn invalid_entry_reports_line_number_and_keeps_nothing() {
        let (_dir, path) =
            write_temp("http://127.0.0.1:9470|chat|tiny-model|deploy-a|1\nnot-an-approval\n");
        let error = read_approvals_file(&path).unwrap_err();
        assert!(
            matches!(error, SharedApprovalsError::InvalidEntry { line: 2, .. }),
            "expected line-2 error, got {error:?}"
        );
    }

    #[test]
    fn missing_file_is_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_approvals_file(&dir.path().join("absent.txt")).unwrap_err();
        assert_eq!(error, SharedApprovalsError::Unreadable);
    }

    #[test]
    fn view_refresh_keeps_previous_approvals_on_failure() {
        let (_dir, path) = write_temp("http://127.0.0.1:9470|chat|tiny-model|deploy-a|1\n");
        let config = SharedApprovalsConfig {
            path: path.clone(),
            ttl: std::time::Duration::from_nanos(1),
        };
        let mut view = SharedApprovalsView::load(config).unwrap();
        assert_eq!(view.approvals().len(), 1);
        std::fs::remove_file(&path).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(view.refresh_if_due().is_err());
        assert_eq!(
            view.approvals().len(),
            1,
            "previous view must survive a failed refresh"
        );
    }

    #[test]
    fn too_many_entries_are_rejected() {
        let mut content = String::new();
        for index in 0..=MAX_SHARED_APPROVAL_ENTRIES {
            content.push_str(&format!(
                "http://127.0.0.1:{}/|chat|tiny-model|deploy-a|1\n",
                9000 + (index % 1000)
            ));
        }
        let (_dir, path) = write_temp(&content);
        assert_eq!(
            read_approvals_file(&path).unwrap_err(),
            SharedApprovalsError::TooManyEntries
        );
    }
}

//! The gateways' shared approvals file, as the supervisor edits it.
//!
//! Autoscaling and rollout both rewrite this file, and an operator or a
//! second supervisor on the same host may edit it too. Every supervisor edit
//! goes through [`edit`]: take the exclusive lock on a sidecar file next to
//! it, read the file under that lock, apply the change to what is there now,
//! write a temporary file, fsync it, rename it over the file and fsync the
//! directory. The sidecar is never removed, so every writer locks the same
//! inode, and because each edit starts from the file as it is under the
//! lock, a writer never loses or resurrects a line another writer changed.
//!
//! Edits are line-preserving: lines a writer does not touch, comments
//! included, pass through byte for byte. Removing exactly the lines an edit
//! appended therefore restores the previous file, which is how a rollout
//! abort undoes its window view without clobbering a peer's change.

use std::io::Write as _;
use std::path::Path;

use fs2::FileExt;
use izwi_serving_protocol::{
    GatewayWorkerApproval, GatewayWorkerApprovalIdentity, NodeId, MAX_APPROVALS_FILE_BYTES,
    MAX_APPROVALS_FILE_ENTRIES,
};
use sha2::{Digest, Sha256};

use crate::WorkerConfig;

#[derive(Debug, thiserror::Error)]
pub enum SharedApprovalsError {
    #[error("approvals file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("approvals file {0}")]
    Invalid(String),
}

impl SharedApprovalsError {
    fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.display().to_string(),
            source,
        }
    }
}

/// One line of the file: its exact bytes (including the newline) and, unless
/// it is blank or a comment, the approval it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ApprovalLine {
    text: String,
    approval: Option<GatewayWorkerApproval>,
}

/// A parsed approvals file that renders back to the same bytes until it is
/// changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalsDocument {
    lines: Vec<ApprovalLine>,
}

impl ApprovalsDocument {
    /// Parses file text. Any invalid entry fails the whole document: the
    /// supervisor never rewrites a file it cannot fully understand.
    pub fn parse(text: &str) -> Result<Self, SharedApprovalsError> {
        if text.len() as u64 > MAX_APPROVALS_FILE_BYTES {
            return Err(SharedApprovalsError::Invalid(format!(
                "exceeds the {MAX_APPROVALS_FILE_BYTES} byte size limit"
            )));
        }
        let mut lines = Vec::new();
        for (index, line) in text.split_inclusive('\n').enumerate() {
            let entry = line.split('#').next().unwrap_or("").trim();
            let approval = if entry.is_empty() {
                None
            } else {
                Some(entry.parse::<GatewayWorkerApproval>().map_err(|error| {
                    SharedApprovalsError::Invalid(format!("line {}: {error}", index + 1))
                })?)
            };
            lines.push(ApprovalLine {
                text: line.to_string(),
                approval,
            });
        }
        Ok(Self { lines })
    }

    /// The approvals in file order.
    pub fn approvals(&self) -> impl Iterator<Item = &GatewayWorkerApproval> {
        self.lines.iter().filter_map(|line| line.approval.as_ref())
    }

    pub fn contains(&self, approval: &GatewayWorkerApproval) -> bool {
        self.approvals().any(|present| present == approval)
    }

    /// Removes every approval line `keep` rejects; blank and comment lines
    /// always stay. Returns the exact text of the removed lines.
    pub fn retain(&mut self, mut keep: impl FnMut(&GatewayWorkerApproval) -> bool) -> Vec<String> {
        let mut removed = Vec::new();
        self.lines.retain(|line| match &line.approval {
            Some(approval) if !keep(approval) => {
                removed.push(line.text.clone());
                false
            }
            _ => true,
        });
        removed
    }

    /// Appends the canonical line for `approval` unless an equal approval is
    /// already present. Returns whether it appended.
    pub fn ensure(&mut self, approval: &GatewayWorkerApproval) -> bool {
        if self.contains(approval) {
            return false;
        }
        self.append_line(&approval.render_line());
        true
    }

    /// Appends one raw line (a line [`retain`](Self::retain) removed earlier).
    pub fn append_line(&mut self, line: &str) {
        if let Some(last) = self.lines.last_mut() {
            if !last.text.ends_with('\n') {
                last.text.push('\n');
            }
        }
        let text = if line.ends_with('\n') {
            line.to_string()
        } else {
            format!("{line}\n")
        };
        let entry = text.split('#').next().unwrap_or("").trim();
        let approval = (!entry.is_empty())
            .then(|| entry.parse::<GatewayWorkerApproval>().ok())
            .flatten();
        self.lines.push(ApprovalLine { text, approval });
    }

    pub fn render(&self) -> String {
        self.lines.iter().map(|line| line.text.as_str()).collect()
    }
}

/// The v1 line pinning one of this node's workers: its loopback endpoint,
/// identity and deployment generation. Autoscaling and rollout both add
/// exactly these lines, so both recognize the lines they own.
pub fn pinned_approval(node_id: &NodeId, worker: &WorkerConfig) -> GatewayWorkerApproval {
    GatewayWorkerApproval {
        endpoint: format!("http://{}", worker.bind),
        identity: GatewayWorkerApprovalIdentity::V1 {
            node_id: node_id.clone(),
            worker_id: worker.worker_id.clone(),
        },
        task: worker.deployment.task,
        public_model: worker.deployment.public_model.clone(),
        deployment_id: worker.deployment.deployment_id.clone(),
        model_generation: worker.deployment.model_generation,
    }
}

/// Reads the file without taking the lock, for preconditions. A missing file
/// is an empty document.
pub fn read(path: &Path) -> Result<ApprovalsDocument, SharedApprovalsError> {
    ApprovalsDocument::parse(&read_text(path)?)
}

fn read_text(path: &Path) -> Result<String, SharedApprovalsError> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(String::new()),
        Err(error) => return Err(SharedApprovalsError::io(path, error)),
    };
    if !metadata.is_file() {
        return Err(SharedApprovalsError::Invalid(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_APPROVALS_FILE_BYTES {
        return Err(SharedApprovalsError::Invalid(format!(
            "{} exceeds the {MAX_APPROVALS_FILE_BYTES} byte size limit",
            path.display()
        )));
    }
    std::fs::read_to_string(path).map_err(|error| SharedApprovalsError::io(path, error))
}

/// Applies `change` to the file under its exclusive lock and returns the
/// digest of the text that is now in place. A missing file starts empty; an
/// unchanged document is not rewritten.
pub fn edit<E>(
    path: &Path,
    change: impl FnOnce(&mut ApprovalsDocument) -> Result<(), E>,
) -> Result<String, E>
where
    E: From<SharedApprovalsError>,
{
    let directory = parent_directory(path)?;
    std::fs::create_dir_all(directory).map_err(|error| SharedApprovalsError::io(path, error))?;
    let lock_path = directory.join(format!(".{}.lock", file_name(path)));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .map_err(|error| SharedApprovalsError::io(&lock_path, error))?;
    lock.lock_exclusive()
        .map_err(|error| SharedApprovalsError::io(&lock_path, error))?;
    // Dropping the file releases the lock; the sidecar itself stays.
    let current = read_text(path)?;
    let mut document = ApprovalsDocument::parse(&current)?;
    change(&mut document)?;
    let text = document.render();
    if text != current {
        if text.len() as u64 > MAX_APPROVALS_FILE_BYTES
            || document.approvals().count() > MAX_APPROVALS_FILE_ENTRIES
        {
            return Err(SharedApprovalsError::Invalid(format!(
                "edit would exceed the gateway's {MAX_APPROVALS_FILE_BYTES} byte or {MAX_APPROVALS_FILE_ENTRIES} entry limit"
            ))
            .into());
        }
        replace_file(path, text.as_bytes())
            .map_err(|error| SharedApprovalsError::io(path, error))?;
    }
    drop(lock);
    Ok(digest(text.as_bytes()))
}

/// Replaces `path` atomically and durably: a temporary file in the same
/// directory (owner-only on unix), fsync, rename, then fsync the directory
/// so the rename itself survives a crash.
pub fn replace_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let directory = path
        .parent()
        .ok_or_else(|| std::io::Error::other(format!("{} has no parent", path.display())))?;
    std::fs::create_dir_all(directory)?;
    let temporary = directory.join(format!(
        ".{}.tmp-{}",
        file_name(path),
        uuid::Uuid::new_v4().simple()
    ));
    let written = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written?;
    // Windows cannot open a directory for fsync; NTFS journals the rename.
    #[cfg(unix)]
    std::fs::File::open(directory)?.sync_all()?;
    Ok(())
}

pub fn digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn parent_directory(path: &Path) -> Result<&Path, SharedApprovalsError> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| {
            SharedApprovalsError::Invalid(format!("{} has no parent directory", path.display()))
        })
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "approvals".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE_A: &str = "v1|http://127.0.0.1:9470|node-a|worker-a|chat|chat-model|chat-prod|1";
    const LINE_B: &str = "v1|http://127.0.0.1:9471|node-a|worker-b|chat|chat-model|chat-prod|1";

    fn approval(line: &str) -> GatewayWorkerApproval {
        line.parse().unwrap()
    }

    #[test]
    fn untouched_lines_and_comments_render_byte_for_byte() {
        let text = format!("# fleet approvals\n{LINE_A}  # pinned\n\n   \n");
        let mut document = ApprovalsDocument::parse(&text).unwrap();
        assert_eq!(document.render(), text);
        assert!(document.ensure(&approval(LINE_B)));
        assert!(!document.ensure(&approval(LINE_B)), "ensure is idempotent");
        let removed = document.retain(|candidate| candidate != &approval(LINE_B));
        assert_eq!(removed, vec![format!("{LINE_B}\n")]);
        assert_eq!(
            document.render(),
            text,
            "removing exactly the appended line restores the file"
        );
    }

    #[test]
    fn invalid_entries_fail_the_whole_document() {
        assert!(ApprovalsDocument::parse(&format!("{LINE_A}\nnot-an-approval\n")).is_err());
    }

    #[test]
    fn edits_are_serialized_and_start_from_the_locked_file() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("approvals");
        std::fs::write(&path, "# shared\n").unwrap();
        // Many writers each add their own line at once. Every edit reads the
        // file under the lock, so no writer can overwrite another's line.
        let writers: Vec<_> = (0..16u16)
            .map(|index| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let line = format!(
                        "v1|http://127.0.0.1:{}|node-a|worker-{index}|chat|chat-model|chat-prod|1",
                        9500 + index
                    );
                    edit::<SharedApprovalsError>(&path, |document| {
                        document.ensure(&approval(&line));
                        Ok(())
                    })
                    .unwrap();
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        let document = read(&path).unwrap();
        assert_eq!(document.approvals().count(), 16, "no edit was lost");
        assert!(
            root.path().join(".approvals.lock").exists(),
            "the lock file stays, so every writer locks the same inode"
        );
        let leftovers: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "no temporary file is left behind");
    }

    #[test]
    fn a_failed_change_leaves_the_file_alone() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("approvals");
        std::fs::write(&path, format!("{LINE_A}\n")).unwrap();
        let result = edit(&path, |document| {
            document.retain(|_| false);
            Err(SharedApprovalsError::Invalid("refused".to_string()))
        });
        assert!(result.is_err());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            format!("{LINE_A}\n")
        );
    }

    #[test]
    fn a_missing_file_starts_empty_and_is_created() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("approvals");
        let digest_written = edit::<SharedApprovalsError>(&path, |document| {
            document.ensure(&approval(LINE_A));
            Ok(())
        })
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, format!("{LINE_A}\n"));
        assert_eq!(digest_written, digest(text.as_bytes()));
    }
}

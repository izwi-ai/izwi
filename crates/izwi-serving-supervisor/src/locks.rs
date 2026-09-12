use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub const MAX_LOCK_METADATA_BYTES: usize = 4096;

#[derive(Debug, Clone)]
pub struct LockNamespace {
    directory: PathBuf,
}

impl LockNamespace {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, LockError> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory).map_err(|source| LockError::Io {
            operation: "create lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
        let metadata = fs::symlink_metadata(directory).map_err(|source| LockError::Io {
            operation: "inspect lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(LockError::UnsafeDirectory(directory.to_path_buf()));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).map_err(
                |source| LockError::Io {
                    operation: "restrict lock directory permissions",
                    path: directory.to_path_buf(),
                    source,
                },
            )?;
        }
        let directory = fs::canonicalize(directory).map_err(|source| LockError::Io {
            operation: "canonicalize lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
        Ok(Self { directory })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn node_supervisor_path(&self) -> PathBuf {
        self.path_for("supervisor", b"node-supervisor")
    }

    pub fn generation_fence_path(&self) -> PathBuf {
        self.path_for("generation", b"worker-generation-fence")
    }

    /// Node-wide serialization point for model construction, loading, and warm-up.
    ///
    /// Resident workers keep their assignment-specific resource lease, but release
    /// this lease as soon as their configured deployment has completed warm-up.
    pub fn model_load_path(&self) -> PathBuf {
        self.path_for("model-load", b"node-model-load-stage")
    }

    pub fn resource_path(&self, resource_identity: &[u8]) -> PathBuf {
        self.path_for("resource", resource_identity)
    }

    pub fn try_node_supervisor(&self, metadata: &[u8]) -> Result<LockLease, LockError> {
        self.try_exclusive(&self.node_supervisor_path(), metadata)
    }

    /// Exclusively fences startup until shared leases from every older worker are gone.
    pub fn try_generation_barrier(&self, metadata: &[u8]) -> Result<LockLease, LockError> {
        self.try_exclusive(&self.generation_fence_path(), metadata)
    }

    pub fn try_exclusive(&self, path: &Path, metadata: &[u8]) -> Result<LockLease, LockError> {
        self.open_and_lock(path, LockMode::Exclusive, metadata, false)
    }

    /// Waits for an exclusive lease.
    ///
    /// This is reserved for bounded supervisor startup stages. The supervising
    /// process owns the child deadline and forcibly tears it down if the stage
    /// cannot make progress before that deadline.
    pub fn lock_exclusive(&self, path: &Path, metadata: &[u8]) -> Result<LockLease, LockError> {
        self.open_and_lock(path, LockMode::Exclusive, metadata, true)
    }

    pub fn try_shared(&self, path: &Path, metadata: &[u8]) -> Result<LockLease, LockError> {
        self.open_and_lock(path, LockMode::Shared, metadata, false)
    }

    fn path_for(&self, prefix: &str, identity: &[u8]) -> PathBuf {
        let digest: [u8; 32] = Sha256::digest(identity).into();
        let mut encoded = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(encoded, "{byte:02x}");
        }
        self.directory.join(format!("{prefix}-{encoded}.lock"))
    }

    fn open_and_lock(
        &self,
        path: &Path,
        mode: LockMode,
        metadata: &[u8],
        wait: bool,
    ) -> Result<LockLease, LockError> {
        if metadata.len() > MAX_LOCK_METADATA_BYTES {
            return Err(LockError::MetadataTooLarge {
                actual: metadata.len(),
                maximum: MAX_LOCK_METADATA_BYTES,
            });
        }
        if path.parent() != Some(self.directory.as_path()) {
            return Err(LockError::PathOutsideNamespace(path.to_path_buf()));
        }
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        }
        let mut file = options.open(path).map_err(|source| LockError::Io {
            operation: "open lock",
            path: path.to_path_buf(),
            source,
        })?;
        let lock_result = match (mode, wait) {
            (LockMode::Exclusive, true) => FileExt::lock_exclusive(&file),
            (LockMode::Exclusive, false) => FileExt::try_lock_exclusive(&file),
            (LockMode::Shared, true) => FileExt::lock_shared(&file),
            (LockMode::Shared, false) => FileExt::try_lock_shared(&file),
        };
        match lock_result {
            Ok(()) => {}
            Err(source) if source.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(LockError::Contended(path.to_path_buf()))
            }
            Err(source) => {
                return Err(LockError::Io {
                    operation: "acquire lock",
                    path: path.to_path_buf(),
                    source,
                })
            }
        }
        if mode == LockMode::Exclusive {
            file.set_len(0).map_err(|source| LockError::Io {
                operation: "truncate lock metadata",
                path: path.to_path_buf(),
                source,
            })?;
            file.seek(SeekFrom::Start(0))
                .map_err(|source| LockError::Io {
                    operation: "seek lock metadata",
                    path: path.to_path_buf(),
                    source,
                })?;
            file.write_all(metadata).map_err(|source| LockError::Io {
                operation: "write lock metadata",
                path: path.to_path_buf(),
                source,
            })?;
            file.sync_data().map_err(|source| LockError::Io {
                operation: "sync lock metadata",
                path: path.to_path_buf(),
                source,
            })?;
        }
        Ok(LockLease {
            file,
            path: path.to_path_buf(),
            mode,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Shared,
    Exclusive,
}

pub struct LockLease {
    file: File,
    path: PathBuf,
    mode: LockMode,
}

/// The two leases a managed worker holds before initializing its runtime.
///
/// The exclusive resource lease prevents duplicate ownership. The shared
/// generation lease prevents a restarted supervisor from launching replacements
/// until every worker from the prior generation has exited.
#[derive(Debug)]
pub struct WorkerFenceLeases {
    pub resource: LockLease,
    pub generation: LockLease,
}

pub fn try_acquire_worker_fences(
    namespace: &LockNamespace,
    resource_path: &Path,
    generation_path: &Path,
    metadata: &[u8],
) -> Result<WorkerFenceLeases, LockError> {
    let resource = namespace.try_exclusive(resource_path, metadata)?;
    let generation = namespace.try_shared(generation_path, b"")?;
    Ok(WorkerFenceLeases {
        resource,
        generation,
    })
}

impl LockLease {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mode(&self) -> LockMode {
        self.mode
    }
}

impl std::fmt::Debug for LockLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LockLease")
            .field("path", &self.path)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl Drop for LockLease {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("unsafe lock directory {0}")]
    UnsafeDirectory(PathBuf),
    #[error("lock path is outside the lock namespace: {0}")]
    PathOutsideNamespace(PathBuf),
    #[error("lock metadata is {actual} bytes; maximum is {maximum}")]
    MetadataTooLarge { actual: usize, maximum: usize },
    #[error("lock is already held: {0}")]
    Contended(PathBuf),
    #[error("failed to {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_names_are_digest_based_and_stable() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        let first = namespace.resource_path(b"cuda:GPU-123/../../unsafe");
        let second = namespace.resource_path(b"cuda:GPU-123/../../unsafe");
        assert_eq!(first, second);
        assert_eq!(first.parent(), Some(namespace.directory()));
        let name = first.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("resource-"));
        assert!(!name.contains("GPU"));
        assert!(!name.contains(".."));
    }

    #[test]
    fn exclusive_lock_contends_and_releases_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        let path = namespace.node_supervisor_path();
        let first = namespace
            .try_exclusive(&path, br#"{"owner":"one"}"#)
            .unwrap();
        assert!(matches!(
            namespace.try_exclusive(&path, br#"{"owner":"two"}"#),
            Err(LockError::Contended(_))
        ));
        drop(first);
        namespace
            .try_exclusive(&path, br#"{"owner":"two"}"#)
            .unwrap();
    }

    #[test]
    fn generation_fence_conflicts_with_live_shared_worker_lease() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        let path = namespace.generation_fence_path();
        let worker = namespace.try_shared(&path, b"").unwrap();
        assert!(matches!(
            namespace.try_exclusive(&path, b"new-supervisor"),
            Err(LockError::Contended(_))
        ));
        drop(worker);
        namespace.try_exclusive(&path, b"new-supervisor").unwrap();
    }

    #[test]
    fn model_load_stage_is_node_wide_and_exclusive() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        let path = namespace.model_load_path();
        let first = namespace.lock_exclusive(&path, b"worker-one").unwrap();
        assert!(matches!(
            namespace.try_exclusive(&path, b"worker-two"),
            Err(LockError::Contended(_))
        ));
        drop(first);
        namespace
            .try_exclusive(&path, b"worker-two")
            .expect("the next load stage starts after the prior stage releases");
    }

    #[test]
    fn partial_worker_fence_acquisition_releases_the_resource_lock() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        let resource = namespace.resource_path(b"cuda:GPU-1234");
        let generation = namespace.generation_fence_path();
        let restart = namespace.try_exclusive(&generation, b"restart").unwrap();
        assert!(matches!(
            try_acquire_worker_fences(&namespace, &resource, &generation, b"worker"),
            Err(LockError::Contended(_))
        ));
        namespace
            .try_exclusive(&resource, b"different-worker")
            .expect("failed fence acquisition must release its resource lease");
        drop(restart);
    }

    #[test]
    fn rejects_paths_and_metadata_outside_bounds() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        assert!(matches!(
            namespace.try_exclusive(&directory.path().join("elsewhere.lock"), b""),
            Err(LockError::PathOutsideNamespace(_))
        ));
        assert!(matches!(
            namespace.try_exclusive(
                &namespace.node_supervisor_path(),
                &vec![b'x'; MAX_LOCK_METADATA_BYTES + 1]
            ),
            Err(LockError::MetadataTooLarge { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_lock_files() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let namespace = LockNamespace::open(directory.path().join("locks")).unwrap();
        let target = directory.path().join("target");
        File::create(&target).unwrap();
        let path = namespace.node_supervisor_path();
        symlink(&target, &path).unwrap();
        assert!(matches!(
            namespace.try_exclusive(&path, b"owner"),
            Err(LockError::Io { .. })
        ));
    }
}

//! Bounded reads of operator-supplied secret and certificate files.
//!
//! Gateways and workers load TLS material and API keys from `file:` references
//! in their environment. Every such read goes through [`read_secret_file`]:
//! the path must be absolute and name a regular file, its size is checked
//! before any byte is read, and the read itself stops one byte past the
//! limit, so a file that grows after the check is still refused. On Unix the
//! result reports a mode that lets other users read the file, so a caller can
//! warn about an exposed private key.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// The prefix of a file reference (`file:/absolute/path`).
pub const FILE_REFERENCE_PREFIX: &str = "file:";

/// The contents of a secret file and how it is shared.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretFile {
    pub bytes: Vec<u8>,
    /// The file's permission bits when its group or other users may access
    /// it (Unix only); `None` when only the owner can.
    pub shared_mode: Option<u32>,
}

impl std::fmt::Debug for SecretFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretFile")
            .field("bytes", &format_args!("[{} bytes]", self.bytes.len()))
            .field("shared_mode", &self.shared_mode)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SecretFileError {
    #[error("the reference must be a file: reference")]
    NotFileReference,
    #[error("the path must be absolute")]
    NotAbsolute,
    #[error("the file does not exist or cannot be opened")]
    Unavailable,
    #[error("the path must name a regular file")]
    NotRegularFile,
    #[error("the file exceeds {0} bytes")]
    TooLarge(u64),
    #[error("the file is empty")]
    Empty,
    #[error("the file could not be read")]
    ReadFailed,
}

/// The absolute path a `file:` reference names.
pub fn file_reference_path(reference: &str) -> Result<&Path, SecretFileError> {
    let path = reference
        .strip_prefix(FILE_REFERENCE_PREFIX)
        .map(Path::new)
        .ok_or(SecretFileError::NotFileReference)?;
    if !path.is_absolute() {
        return Err(SecretFileError::NotAbsolute);
    }
    Ok(path)
}

/// Reads at most `max_bytes` from the regular file at the absolute `path`.
pub fn read_secret_file(path: &Path, max_bytes: u64) -> Result<SecretFile, SecretFileError> {
    if !path.is_absolute() {
        return Err(SecretFileError::NotAbsolute);
    }
    // Check the type before opening: opening a FIFO would block.
    let metadata = std::fs::metadata(path).map_err(|_| SecretFileError::Unavailable)?;
    if !metadata.is_file() {
        return Err(SecretFileError::NotRegularFile);
    }
    let file = File::open(path).map_err(|_| SecretFileError::Unavailable)?;
    // The opened file is the one read; recheck it in case the path changed.
    let metadata = file.metadata().map_err(|_| SecretFileError::ReadFailed)?;
    if !metadata.is_file() {
        return Err(SecretFileError::NotRegularFile);
    }
    if metadata.len() > max_bytes {
        return Err(SecretFileError::TooLarge(max_bytes));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| SecretFileError::ReadFailed)?;
    if bytes.len() as u64 > max_bytes {
        return Err(SecretFileError::TooLarge(max_bytes));
    }
    if bytes.is_empty() {
        return Err(SecretFileError::Empty);
    }
    Ok(SecretFile {
        bytes,
        shared_mode: shared_mode(&metadata),
    })
}

#[cfg(unix)]
fn shared_mode(metadata: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    let mode = metadata.permissions().mode() & 0o777;
    (mode & 0o077 != 0).then_some(mode)
}

#[cfg(not(unix))]
fn shared_mode(_metadata: &std::fs::Metadata) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A scratch directory removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "izwi-secret-file-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, name: &str, contents: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, contents).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            }
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn file_references_must_be_absolute_file_references() {
        assert_eq!(
            file_reference_path("file:/etc/izwi/key.pem").unwrap(),
            Path::new("/etc/izwi/key.pem")
        );
        assert_eq!(
            file_reference_path("file:key.pem").unwrap_err(),
            SecretFileError::NotAbsolute
        );
        assert_eq!(
            file_reference_path("env:KEY").unwrap_err(),
            SecretFileError::NotFileReference
        );
    }

    #[test]
    fn reads_a_bounded_owner_only_file() {
        let scratch = Scratch::new();
        let path = scratch.file("key", b"secret");
        let file = read_secret_file(&path, 6).unwrap();
        assert_eq!(file.bytes, b"secret");
        assert_eq!(file.shared_mode, None);
    }

    #[test]
    fn refuses_relative_missing_empty_and_non_regular_paths() {
        let scratch = Scratch::new();
        assert_eq!(
            read_secret_file(Path::new("key"), 64).unwrap_err(),
            SecretFileError::NotAbsolute
        );
        assert_eq!(
            read_secret_file(&scratch.0.join("missing"), 64).unwrap_err(),
            SecretFileError::Unavailable
        );
        assert_eq!(
            read_secret_file(&scratch.file("empty", b""), 64).unwrap_err(),
            SecretFileError::Empty
        );
        assert_eq!(
            read_secret_file(&scratch.0, 64).unwrap_err(),
            SecretFileError::NotRegularFile
        );
    }

    #[test]
    fn an_oversized_file_is_refused() {
        let scratch = Scratch::new();
        assert_eq!(
            read_secret_file(&scratch.file("seven", b"1234567"), 6).unwrap_err(),
            SecretFileError::TooLarge(6)
        );
    }

    /// A sparse 1 TiB file (Unix filesystems allocate it lazily): refusing it
    /// must not read it.
    #[cfg(unix)]
    #[test]
    fn an_oversized_file_is_refused_by_its_size_before_reading() {
        let scratch = Scratch::new();
        let path = scratch.file("huge", b"");
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1 << 40)
            .unwrap();
        assert_eq!(
            read_secret_file(&path, 64).unwrap_err(),
            SecretFileError::TooLarge(64)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_refused_without_blocking() {
        let scratch = Scratch::new();
        let fifo = scratch.0.join("fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            read_secret_file(&fifo, 64).unwrap_err(),
            SecretFileError::NotRegularFile
        );
    }

    #[cfg(unix)]
    #[test]
    fn reports_a_file_other_users_can_read() {
        use std::os::unix::fs::PermissionsExt;
        let scratch = Scratch::new();
        let path = scratch.file("key", b"secret");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            read_secret_file(&path, 64).unwrap().shared_mode,
            Some(0o644)
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        assert_eq!(
            read_secret_file(&path, 64).unwrap().shared_mode,
            Some(0o640)
        );
    }
}

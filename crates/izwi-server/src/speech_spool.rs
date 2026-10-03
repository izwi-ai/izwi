//! Process-owned scratch files for local speech generation.
//!
//! Cleanup relies only on an exclusive ownership lock. It deliberately does
//! not infer liveness from a PID, timestamp or lease.

use anyhow::{anyhow, Context};
use fs2::FileExt;
use std::fs::{DirEntry, File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, OnceLock,
};

const OWNER_LOCK_NAME: &str = ".owner.lock";
const TEMP_PREFIX: &str = "speech-";
const TEMP_SUFFIX: &str = ".tmp";
const TEMP_RANDOM_CHARS: usize = 12;
const MAX_ROOT_ENTRIES: usize = 256;
const MAX_OWNER_ENTRIES: usize = 256;
const MAX_PROCESS_TEMPFILES: usize = MAX_OWNER_ENTRIES - 1;
const OWNER_CREATE_ATTEMPTS: usize = 8;
const MAX_SPOOL_PATH_BYTES: usize = 4 * 1024;

static PROCESS_SPOOL: OnceLock<Result<Arc<SpeechSpool>, String>> = OnceLock::new();

/// Create a local speech temporary file lazily. Starting a gateway or a local
/// server that never needs speech scratch space does not touch this directory.
pub(crate) fn new_speech_tempfile() -> anyhow::Result<SpeechTempFile> {
    let owner = PROCESS_SPOOL.get_or_init(|| {
        SpeechSpool::initialize(crate::storage_layout::resolve_speech_spool_root())
            .map(Arc::new)
            .map_err(|error| format!("Initialize process-owned speech spool: {error:#}"))
    });
    match owner {
        Ok(owner) => owner.new_tempfile(),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

struct SpeechSpool {
    owner_dir: PathBuf,
    #[allow(dead_code)]
    owner_lock: File,
    active_tempfiles: AtomicUsize,
}

/// Keeps both the temporary file and its process ownership slot alive. The
/// file is removed before the slot is released, so normal operation can never
/// create more entries than startup recovery is prepared to inspect.
pub(crate) struct SpeechTempFile {
    temporary: Option<tempfile::NamedTempFile>,
    owner: Arc<SpeechSpool>,
}

impl SpeechTempFile {
    pub(crate) fn path(&self) -> &Path {
        self.temporary
            .as_ref()
            .expect("speech temporary file remains present until drop")
            .path()
    }

    pub(crate) fn reopen(&self) -> std::io::Result<File> {
        self.temporary
            .as_ref()
            .expect("speech temporary file remains present until drop")
            .reopen()
    }
}

impl Drop for SpeechTempFile {
    fn drop(&mut self) {
        let release_slot = match self.temporary.take() {
            Some(temporary) => match temporary.close() {
                Ok(()) => true,
                Err(error) if error.kind() == ErrorKind::NotFound => true,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "Speech temporary file could not be removed; retaining its capacity slot"
                    );
                    false
                }
            },
            None => false,
        };
        if release_slot {
            self.owner.active_tempfiles.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl SpeechSpool {
    fn initialize(root: PathBuf) -> anyhow::Result<Self> {
        validate_path_bound(&root)?;
        ensure_private_directory(&root)
            .with_context(|| format!("Prepare speech spool root '{}'", root.display()))?;
        let root = std::fs::canonicalize(&root)
            .with_context(|| format!("Canonicalize speech spool root '{}'", root.display()))?;
        validate_path_bound(&root)?;
        ensure_private_directory(&root)?;
        scavenge_stale_owners(&root)?;

        let remaining = read_entries_bounded(&root, MAX_ROOT_ENTRIES)?;
        anyhow::ensure!(
            remaining.len() < MAX_ROOT_ENTRIES,
            "Speech spool root contains the maximum of {MAX_ROOT_ENTRIES} entries"
        );

        for _ in 0..OWNER_CREATE_ATTEMPTS {
            let owner_dir = root.join(uuid::Uuid::new_v4().hyphenated().to_string());
            match create_private_directory(&owner_dir) {
                Ok(()) => match ensure_private_directory(&owner_dir)
                    .and_then(|()| create_and_lock_owner(&owner_dir))
                {
                    Ok(owner_lock) => {
                        return Ok(Self {
                            owner_dir,
                            owner_lock,
                            active_tempfiles: AtomicUsize::new(0),
                        });
                    }
                    Err(error) => {
                        let _ = std::fs::remove_dir_all(&owner_dir);
                        return Err(error);
                    }
                },
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "Create speech spool owner directory '{}'",
                            owner_dir.display()
                        )
                    });
                }
            }
        }

        Err(anyhow!(
            "Could not allocate a unique process-owned speech spool directory"
        ))
    }

    fn new_tempfile(self: &Arc<Self>) -> anyhow::Result<SpeechTempFile> {
        self.active_tempfiles
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < MAX_PROCESS_TEMPFILES).then_some(active + 1)
            })
            .map_err(|_| {
                anyhow!("Speech spool reached its {MAX_PROCESS_TEMPFILES}-temporary-file capacity")
            })?;
        let temporary = match tempfile::Builder::new()
            .prefix(TEMP_PREFIX)
            .suffix(TEMP_SUFFIX)
            .rand_bytes(TEMP_RANDOM_CHARS)
            .tempfile_in(&self.owner_dir)
        {
            Ok(temporary) => temporary,
            Err(error) => {
                self.active_tempfiles.fetch_sub(1, Ordering::AcqRel);
                return Err(error).with_context(|| {
                    format!(
                        "Create temporary speech file in process owner directory '{}'",
                        self.owner_dir.display()
                    )
                });
            }
        };
        Ok(SpeechTempFile {
            temporary: Some(temporary),
            owner: Arc::clone(self),
        })
    }
}

fn validate_path_bound(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.as_os_str().as_encoded_bytes().len() <= MAX_SPOOL_PATH_BYTES,
        "Speech spool path exceeds {MAX_SPOOL_PATH_BYTES} bytes"
    );
    Ok(())
}

fn scavenge_stale_owners(root: &Path) -> anyhow::Result<()> {
    let entries = read_entries_bounded(root, MAX_ROOT_ENTRIES)?;
    for entry in entries {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(uuid) = uuid::Uuid::parse_str(&name) else {
            continue;
        };
        if uuid.hyphenated().to_string() != name {
            continue;
        }
        let path = entry.path();
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            continue;
        }
        // A malformed or changing sibling is left untouched. Scratch cleanup
        // must never turn an uncertain ownership state into data loss.
        let _ = try_remove_stale_owner(&path);
    }
    Ok(())
}

fn try_remove_stale_owner(owner_dir: &Path) -> anyhow::Result<bool> {
    let entries = match read_entries_bounded(owner_dir, MAX_OWNER_ENTRIES) {
        Ok(entries) => entries,
        Err(_) => return Ok(false),
    };
    let mut lock_path = None;
    let mut temp_paths = Vec::new();
    for entry in entries {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Ok(false);
        };
        let metadata = match std::fs::symlink_metadata(entry.path()) {
            Ok(metadata) => metadata,
            Err(_) => return Ok(false),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Ok(false);
        }
        if name == OWNER_LOCK_NAME {
            if lock_path.replace(entry.path()).is_some() {
                return Ok(false);
            }
        } else if is_recognized_temp_name(&name) {
            temp_paths.push(entry.path());
        } else {
            return Ok(false);
        }
    }

    let Some(lock_path) = lock_path else {
        return Ok(false);
    };
    let lock_file = match OpenOptions::new().read(true).write(true).open(&lock_path) {
        Ok(file) => file,
        Err(_) => return Ok(false),
    };
    match FileExt::try_lock_exclusive(&lock_file) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(false),
        Err(_) => return Ok(false),
    }

    for path in temp_paths {
        if let Err(error) = std::fs::remove_file(&path) {
            if error.kind() != ErrorKind::NotFound {
                return Ok(false);
            }
        }
    }
    // The exact lock established that no owner is live. Release the handle
    // before unlinking it so cleanup also works on platforms that prohibit
    // deleting open files. UUID directories are never adopted by a process.
    if FileExt::unlock(&lock_file).is_err() {
        return Ok(false);
    }
    drop(lock_file);
    if let Err(error) = std::fs::remove_file(&lock_path) {
        if error.kind() != ErrorKind::NotFound {
            return Ok(false);
        }
    }
    match std::fs::remove_dir(owner_dir) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(true),
        Err(_) => Ok(false),
    }
}

fn create_and_lock_owner(owner_dir: &Path) -> anyhow::Result<File> {
    let lock_path = owner_dir.join(OWNER_LOCK_NAME);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .with_context(|| format!("Create speech spool lock '{}'", lock_path.display()))?;
    FileExt::try_lock_exclusive(&file)
        .with_context(|| format!("Lock speech spool owner '{}'", lock_path.display()))?;
    Ok(file)
}

fn read_entries_bounded(path: &Path, limit: usize) -> anyhow::Result<Vec<DirEntry>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("Read speech spool directory '{}'", path.display()))?
    {
        anyhow::ensure!(
            entries.len() < limit,
            "Speech spool directory '{}' exceeds its {limit}-entry scan limit",
            path.display()
        );
        entries.push(
            entry.with_context(|| format!("Read speech spool entry below '{}'", path.display()))?,
        );
    }
    Ok(entries)
}

fn is_recognized_temp_name(name: &str) -> bool {
    let Some(random) = name
        .strip_prefix(TEMP_PREFIX)
        .and_then(|name| name.strip_suffix(TEMP_SUFFIX))
    else {
        return false;
    };
    random.len() == TEMP_RANDOM_CHARS && random.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn ensure_private_directory(path: &Path) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("Create speech spool parent '{}'", parent.display())
                })?;
            }
            match create_private_directory(path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(error) => return Err(error.into()),
    }
    let metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        "Speech spool root must be a non-symlink directory"
    );
    ensure_owner_only_permissions(path, &metadata)
}

#[cfg(unix)]
fn create_private_directory(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_directory(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir(path)
}

#[cfg(unix)]
fn ensure_owner_only_permissions(path: &Path, metadata: &std::fs::Metadata) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    anyhow::ensure!(
        metadata.permissions().mode() & 0o077 == 0,
        "Speech spool directory '{}' must not grant group or other permissions",
        path.display()
    );
    Ok(())
}

#[cfg(windows)]
fn ensure_owner_only_permissions(path: &Path, _metadata: &std::fs::Metadata) -> anyhow::Result<()> {
    use std::mem::size_of;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::{
        Foundation::{CloseHandle, ERROR_SUCCESS, HANDLE},
        Security::{
            AddAccessAllowedAceEx,
            Authorization::{SetNamedSecurityInfoW, SE_FILE_OBJECT},
            GetLengthSid, GetTokenInformation, InitializeAcl, TokenUser, ACCESS_ALLOWED_ACE, ACL,
            ACL_REVISION, CONTAINER_INHERIT_ACE, DACL_SECURITY_INFORMATION, OBJECT_INHERIT_ACE,
            PROTECTED_DACL_SECURITY_INFORMATION, TOKEN_QUERY, TOKEN_USER,
        },
        Storage::FileSystem::FILE_ALL_ACCESS,
        System::Threading::{GetCurrentProcess, OpenProcessToken},
    };

    struct TokenHandle(HANDLE);
    impl Drop for TokenHandle {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    let mut token = std::ptr::null_mut();
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(std::io::Error::last_os_error()).context("Open current process token");
    }
    let token = TokenHandle(token);

    let mut token_bytes = 0u32;
    unsafe {
        GetTokenInformation(
            token.0,
            TokenUser,
            std::ptr::null_mut(),
            0,
            &mut token_bytes,
        );
    }
    anyhow::ensure!(token_bytes > 0, "Determine current Windows token user size");
    let word = size_of::<usize>();
    let mut token_buffer = vec![0usize; (token_bytes as usize).div_ceil(word)];
    if unsafe {
        GetTokenInformation(
            token.0,
            TokenUser,
            token_buffer.as_mut_ptr().cast(),
            token_bytes,
            &mut token_bytes,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error()).context("Read current Windows token user");
    }
    let user_sid = unsafe { (*(token_buffer.as_ptr().cast::<TOKEN_USER>())).User.Sid };
    let sid_bytes = unsafe { GetLengthSid(user_sid) } as usize;
    anyhow::ensure!(sid_bytes > 0, "Read current Windows user SID length");

    let acl_bytes = size_of::<ACL>()
        .checked_add(size_of::<ACCESS_ALLOWED_ACE>() - size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(sid_bytes))
        .context("Windows speech spool ACL size overflow")?;
    let mut acl_buffer = vec![0u32; acl_bytes.div_ceil(size_of::<u32>())];
    let acl = acl_buffer.as_mut_ptr().cast::<ACL>();
    if unsafe { InitializeAcl(acl, acl_bytes as u32, ACL_REVISION) } == 0 {
        return Err(std::io::Error::last_os_error()).context("Initialize speech spool ACL");
    }
    if unsafe {
        AddAccessAllowedAceEx(
            acl,
            ACL_REVISION,
            OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE,
            FILE_ALL_ACCESS,
            user_sid,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error())
            .context("Grant speech spool access to current Windows user");
    }

    let mut wide_path: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide_path.push(0);
    let result = unsafe {
        SetNamedSecurityInfoW(
            wide_path.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            acl,
            std::ptr::null(),
        )
    };
    anyhow::ensure!(
        result == ERROR_SUCCESS,
        "Restrict Windows speech spool ACL failed with error {result}"
    );
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn ensure_owner_only_permissions(
    _path: &Path,
    _metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    anyhow::bail!("Owner-only speech spool permissions are unsupported on this platform")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn private_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        root
    }

    fn create_owner(root: &Path) -> (PathBuf, File) {
        let dir = root.join(uuid::Uuid::new_v4().hyphenated().to_string());
        create_private_directory(&dir).unwrap();
        let lock = create_and_lock_owner(&dir).unwrap();
        (dir, lock)
    }

    #[test]
    fn stale_owner_is_removed_but_live_lock_survives() {
        let root = private_root();
        let (stale, stale_lock) = create_owner(root.path());
        std::fs::write(stale.join("speech-abcdefghijkl.tmp"), b"partial").unwrap();
        FileExt::unlock(&stale_lock).unwrap();
        drop(stale_lock);

        let (live, live_lock) = create_owner(root.path());
        std::fs::write(live.join("speech-abcdefghijkl.tmp"), b"active").unwrap();
        let owner = SpeechSpool::initialize(root.path().to_path_buf()).unwrap();

        assert!(!stale.exists());
        assert!(live.exists());
        assert!(owner.owner_dir.exists());
        FileExt::unlock(&live_lock).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_and_unrecognized_entries_survive_scavenging() {
        use std::os::unix::fs::symlink;
        let root = private_root();
        let target = root.path().join("target");
        std::fs::create_dir(&target).unwrap();
        let linked_uuid = root
            .path()
            .join(uuid::Uuid::new_v4().hyphenated().to_string());
        symlink(&target, &linked_uuid).unwrap();

        let (unexpected, unexpected_lock) = create_owner(root.path());
        FileExt::unlock(&unexpected_lock).unwrap();
        drop(unexpected_lock);
        std::fs::write(unexpected.join("do-not-delete"), b"valuable").unwrap();

        let outside_file = root.path().join("outside-file");
        std::fs::write(&outside_file, b"outside").unwrap();
        let (linked_entry, linked_entry_lock) = create_owner(root.path());
        FileExt::unlock(&linked_entry_lock).unwrap();
        drop(linked_entry_lock);
        symlink(&outside_file, linked_entry.join("speech-abcdefghijkl.tmp")).unwrap();

        let owner = SpeechSpool::initialize(root.path().to_path_buf()).unwrap();
        assert!(std::fs::symlink_metadata(linked_uuid)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(unexpected.join("do-not-delete").exists());
        assert!(linked_entry.join("speech-abcdefghijkl.tmp").exists());
        assert_eq!(std::fs::read(outside_file).unwrap(), b"outside");
        assert!(owner.owner_dir.exists());
    }

    #[test]
    fn root_and_owner_scan_caps_fail_closed() {
        let root = private_root();
        for index in 0..MAX_ROOT_ENTRIES {
            std::fs::write(root.path().join(format!("unknown-{index}")), b"").unwrap();
        }
        assert!(SpeechSpool::initialize(root.path().to_path_buf()).is_err());

        let bounded_root = private_root();
        let (stale, stale_lock) = create_owner(bounded_root.path());
        FileExt::unlock(&stale_lock).unwrap();
        drop(stale_lock);
        for index in 0..MAX_OWNER_ENTRIES {
            std::fs::write(stale.join(format!("speech-{index:012}.tmp")), b"").unwrap();
        }
        let owner = SpeechSpool::initialize(bounded_root.path().to_path_buf()).unwrap();
        assert!(stale.exists());
        assert!(owner.owner_dir.exists());
    }

    #[test]
    fn tempfile_is_in_owner_directory_and_drop_removes_it() {
        let root = private_root();
        let owner = Arc::new(SpeechSpool::initialize(root.path().to_path_buf()).unwrap());
        let temporary = owner.new_tempfile().unwrap();
        let path = temporary.path().to_path_buf();
        assert_eq!(path.parent(), Some(owner.owner_dir.as_path()));
        assert!(path.exists());
        drop(temporary);
        assert!(!path.exists());
    }

    #[test]
    fn normal_creation_is_capped_at_the_recoverable_entry_count() {
        let root = private_root();
        let owner = Arc::new(SpeechSpool::initialize(root.path().to_path_buf()).unwrap());
        let mut temporary = Vec::new();
        for _ in 0..MAX_PROCESS_TEMPFILES {
            temporary.push(owner.new_tempfile().unwrap());
        }
        assert_eq!(
            read_entries_bounded(&owner.owner_dir, MAX_OWNER_ENTRIES)
                .unwrap()
                .len(),
            MAX_OWNER_ENTRIES
        );
        assert!(owner.new_tempfile().is_err());

        drop(temporary.pop());
        temporary.push(owner.new_tempfile().unwrap());
        assert_eq!(temporary.len(), MAX_PROCESS_TEMPFILES);
    }

    #[cfg(unix)]
    #[test]
    fn failed_unlink_retains_the_capacity_slot() {
        use std::os::unix::fs::PermissionsExt;
        let root = private_root();
        let owner = Arc::new(SpeechSpool::initialize(root.path().to_path_buf()).unwrap());
        let temporary = owner.new_tempfile().unwrap();
        let path = temporary.path().to_path_buf();

        std::fs::set_permissions(&owner.owner_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        drop(temporary);
        assert!(path.exists());
        assert_eq!(owner.active_tempfiles.load(Ordering::Acquire), 1);

        std::fs::set_permissions(&owner.owner_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn spool_root_must_be_private_and_not_a_symlink() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let base = tempfile::tempdir().unwrap();
        let broad = base.path().join("broad");
        std::fs::create_dir(&broad).unwrap();
        std::fs::set_permissions(&broad, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(SpeechSpool::initialize(broad).is_err());

        let target = base.path().join("target-root");
        create_private_directory(&target).unwrap();
        let linked = base.path().join("linked-root");
        symlink(target, &linked).unwrap();
        assert!(SpeechSpool::initialize(linked).is_err());
    }

    #[test]
    fn configured_path_length_is_bounded_before_filesystem_access() {
        let oversized = PathBuf::from("a".repeat(MAX_SPOOL_PATH_BYTES + 1));
        let error = match SpeechSpool::initialize(oversized) {
            Ok(_) => panic!("oversized speech spool path was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("exceeds"));
    }
}

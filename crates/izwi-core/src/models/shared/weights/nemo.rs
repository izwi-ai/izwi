//! NeMo `.nemo` archives (a tar, sometimes gzipped) and the disk they need
//! (PKT-9, NEM-15, ADM-14). Entries stream straight to their files, so an
//! extraction holds one copy buffer of host memory however large the
//! checkpoint, and every extraction first checks that its volume has room.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use flate2::read::GzDecoder;
use tar::Archive;

use crate::error::{Error, Result};

/// Free space an extraction or download leaves on its volume beyond what it
/// writes, so a model artifact never fills the disk the OS and the server
/// need for swap, logs and temporary files.
pub(crate) const DISK_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

/// Fail unless `available_bytes` covers `required_bytes` plus
/// [`DISK_HEADROOM_BYTES`].
pub(crate) fn check_free_disk(
    available_bytes: u64,
    required_bytes: u64,
    location: &Path,
    purpose: &str,
) -> Result<()> {
    let needed = required_bytes.saturating_add(DISK_HEADROOM_BYTES);
    if available_bytes < needed {
        return Err(Error::ModelLoadError(format!(
            "Not enough free disk space for {purpose} at {}: required_bytes={required_bytes}, headroom_bytes={DISK_HEADROOM_BYTES}, available_bytes={available_bytes}; free up disk space and retry",
            location.display()
        )));
    }
    Ok(())
}

/// [`check_free_disk`] against the volume holding `location` (statvfs). A
/// volume that cannot be queried is reported and not treated as full.
pub(crate) fn ensure_free_disk(location: &Path, required_bytes: u64, purpose: &str) -> Result<()> {
    match fs2::available_space(location) {
        Ok(available) => check_free_disk(available, required_bytes, location, purpose),
        Err(error) => {
            tracing::warn!(
                location = %location.display(),
                %error,
                "Could not read free disk space; continuing without the check"
            );
            Ok(())
        }
    }
}

/// Open a `.nemo` archive, gunzipping it when it carries the gzip magic.
pub(crate) fn open_nemo_archive(nemo_path: &Path) -> Result<Archive<Box<dyn Read>>> {
    let mut file = File::open(nemo_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed opening .nemo archive {}: {}",
            nemo_path.display(),
            e
        ))
    })?;
    let mut magic = [0u8; 2];
    let read = file.read(&mut magic).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed reading .nemo archive header {}: {}",
            nemo_path.display(),
            e
        ))
    })?;
    file.seek(SeekFrom::Start(0)).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed rewinding .nemo archive {}: {}",
            nemo_path.display(),
            e
        ))
    })?;

    let reader: Box<dyn Read> = if read == 2 && magic == [0x1f, 0x8b] {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Archive::new(reader))
}

/// Stream one archive entry to `dest` through a temporary file, after
/// checking that the destination volume has room for it.
pub(crate) fn extract_entry_to_file<R: Read>(
    entry: &mut tar::Entry<'_, R>,
    dest: &Path,
) -> Result<()> {
    let directory = dest.parent().unwrap_or_else(|| Path::new("."));
    ensure_free_disk(
        directory,
        entry.size(),
        &format!("extracting {}", dest.display()),
    )?;
    let tmp_path = dest.with_extension("tmp");
    let mut tmp_file = File::create(&tmp_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed creating temp extraction file {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    io::copy(entry, &mut tmp_file).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed writing extracted .nemo entry to {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    fs::rename(&tmp_path, dest).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed moving extracted artifact into {}: {}",
            dest.display(),
            e
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_disk_check_requires_the_bytes_and_the_headroom() {
        let location = Path::new("/models");
        assert!(check_free_disk(10 + DISK_HEADROOM_BYTES, 10, location, "test").is_ok());
        let error = check_free_disk(9 + DISK_HEADROOM_BYTES, 10, location, "extracting x")
            .unwrap_err()
            .to_string();
        assert!(error.contains("extracting x"), "{error}");
        assert!(error.contains("required_bytes=10"), "{error}");
        assert!(check_free_disk(u64::MAX, u64::MAX, location, "test").is_ok());
    }
}

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

#[derive(Debug, Clone)]
pub struct SortformerArtifacts {
    pub nemo_path: PathBuf,
    pub extracted_dir: PathBuf,
    pub model_config_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

/// Where `ensure_sortformer_artifacts` finds the `.nemo` archive and puts the
/// files it extracts, computed without touching the filesystem. Load
/// admission reads the checkpoint header (`runtime/lifecycle/asr_memory.rs`).
#[derive(Debug, Clone)]
pub(crate) struct SortformerCheckpointPaths {
    pub(crate) nemo_path: PathBuf,
    pub(crate) extracted_dir: PathBuf,
    pub(crate) checkpoint_path: PathBuf,
}

pub(crate) fn sortformer_checkpoint_paths(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<SortformerCheckpointPaths> {
    let nemo_filename = match variant {
        ModelVariant::DiarStreamingSortformer4SpkV21 => "diar_streaming_sortformer_4spk-v2.1.nemo",
        ModelVariant::Nemotron3Diarization => "Nemotron-3-Diarization.nemo",
        _ => {
            return Err(Error::InvalidInput(format!(
                "Unsupported Sortformer diarization variant: {}",
                variant.dir_name()
            )));
        }
    };
    let extracted_dir = model_dir.join("sortformer-native");
    Ok(SortformerCheckpointPaths {
        nemo_path: model_dir.join(nemo_filename),
        checkpoint_path: extracted_dir.join("model_weights.ckpt"),
        extracted_dir,
    })
}

/// Whether the extracted checkpoint is the one the next load serves: its
/// stamp records the archive's current size and modification time. Reads
/// metadata only; a load settles a changed time by digest, and admission
/// prices anything else as a first load.
pub(crate) fn sortformer_extraction_is_current(paths: &SortformerCheckpointPaths) -> bool {
    let Some(stamp) = ExtractionStamp::read(&paths.extracted_dir.join(EXTRACTION_STAMP_FILE))
    else {
        return false;
    };
    paths.checkpoint_path.is_file()
        && ExtractionStamp::of(&paths.nemo_path, None)
            .is_ok_and(|current| stamp.same_file_metadata(&current))
}

pub fn ensure_sortformer_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<SortformerArtifacts> {
    let SortformerCheckpointPaths {
        nemo_path,
        extracted_dir,
        checkpoint_path,
    } = sortformer_checkpoint_paths(model_dir, variant)?;
    if !nemo_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing .nemo checkpoint for {} at {}",
            variant.dir_name(),
            nemo_path.display()
        )));
    }

    fs::create_dir_all(&extracted_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create Sortformer cache directory {}: {}",
            extracted_dir.display(),
            e
        ))
    })?;

    let model_config_path = extracted_dir.join("model_config.yaml");
    let stamp_path = extracted_dir.join(EXTRACTION_STAMP_FILE);
    let extracted = model_config_path.exists() && checkpoint_path.exists();
    if !extracted || !extraction_matches_archive(&nemo_path, &stamp_path)? {
        // A load interrupted mid-extraction finds no stamp and starts over.
        remove_stamp(&stamp_path)?;
        let stamp = ExtractionStamp::of(&nemo_path, Some(sha256_file(&nemo_path)?))?;
        extract_required_entries(
            &nemo_path,
            &[
                ("model_config.yaml", &model_config_path),
                ("model_weights.ckpt", &checkpoint_path),
            ],
        )?;
        stamp.write(&stamp_path)?;
    }

    Ok(SortformerArtifacts {
        nemo_path,
        extracted_dir,
        model_config_path,
        checkpoint_path,
    })
}

/// Records which `.nemo` the files in `sortformer-native/` came from (NDZ-5),
/// so a re-published archive is extracted again instead of serving the
/// stale config and weights.
const EXTRACTION_STAMP_FILE: &str = "source.nemo.json";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ExtractionStamp {
    size: u64,
    modified_secs: u64,
    modified_nanos: u32,
    sha256: Option<String>,
}

impl ExtractionStamp {
    /// The archive's size and modification time, and its digest when known.
    fn of(nemo_path: &Path, sha256: Option<String>) -> Result<Self> {
        let metadata = fs::metadata(nemo_path).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed reading .nemo archive metadata {}: {}",
                nemo_path.display(),
                e
            ))
        })?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .unwrap_or_default();
        Ok(Self {
            size: metadata.len(),
            modified_secs: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
            sha256,
        })
    }

    fn read(path: &Path) -> Option<Self> {
        serde_json::from_slice(&fs::read(path).ok()?).ok()
    }

    fn write(&self, path: &Path) -> Result<()> {
        let tmp_path = path.with_extension("tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(&tmp_path, bytes)
            .and_then(|()| fs::rename(&tmp_path, path))
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed writing Sortformer extraction stamp {}: {}",
                    path.display(),
                    e
                ))
            })
    }

    fn same_file_metadata(&self, other: &Self) -> bool {
        self.size == other.size
            && self.modified_secs == other.modified_secs
            && self.modified_nanos == other.modified_nanos
    }
}

/// Whether the extraction came from the archive now at `nemo_path`. The
/// metadata check costs nothing per load; a changed modification time with an
/// unchanged size (a copy or touch) is settled by the digest recorded at
/// extraction, and a missing stamp (an extraction from before stamps) counts
/// as stale.
fn extraction_matches_archive(nemo_path: &Path, stamp_path: &Path) -> Result<bool> {
    let Some(stamp) = ExtractionStamp::read(stamp_path) else {
        return Ok(false);
    };
    let current = ExtractionStamp::of(nemo_path, None)?;
    if stamp.same_file_metadata(&current) {
        return Ok(true);
    }
    let Some(recorded) = stamp.sha256.as_deref() else {
        return Ok(false);
    };
    if stamp.size != current.size {
        return Ok(false);
    }
    let digest = sha256_file(nemo_path)?;
    if digest != recorded {
        return Ok(false);
    }
    ExtractionStamp {
        sha256: Some(digest),
        ..current
    }
    .write(stamp_path)?;
    Ok(true)
}

fn remove_stamp(stamp_path: &Path) -> Result<()> {
    match fs::remove_file(stamp_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::ModelLoadError(format!(
            "Failed removing Sortformer extraction stamp {}: {}",
            stamp_path.display(),
            e
        ))),
    }
}

/// The archive's sha256, as Hugging Face publishes it for the LFS file.
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|e| Error::ModelLoadError(format!("Failed opening {}: {}", path.display(), e)))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)
        .map_err(|e| Error::ModelLoadError(format!("Failed hashing {}: {}", path.display(), e)))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_required_entries(nemo_path: &Path, entries: &[(&str, &Path)]) -> Result<()> {
    let mut archive = open_nemo_archive(nemo_path)?;
    let mut extracted = vec![false; entries.len()];

    for entry in archive.entries().map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed reading .nemo archive {}: {}",
            nemo_path.display(),
            e
        ))
    })? {
        let mut entry = entry.map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed iterating .nemo archive {}: {}",
                nemo_path.display(),
                e
            ))
        })?;
        let entry_path = entry
            .path()
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading archive entry path in {}: {}",
                    nemo_path.display(),
                    e
                ))
            })?
            .to_string_lossy()
            .into_owned();

        for (idx, (suffix, dest)) in entries.iter().enumerate() {
            if extracted[idx] || !entry_path.ends_with(suffix) {
                continue;
            }
            extract_entry_to_file(&mut entry, dest)?;
            extracted[idx] = true;
        }

        if extracted.iter().all(|done| *done) {
            return Ok(());
        }
    }

    let missing = entries
        .iter()
        .enumerate()
        .filter_map(|(idx, (suffix, _))| (!extracted[idx]).then_some(*suffix))
        .collect::<Vec<_>>();
    Err(Error::ModelLoadError(format!(
        "Missing required files in .nemo archive {}: {}",
        nemo_path.display(),
        missing.join(", ")
    )))
}

fn open_nemo_archive(nemo_path: &Path) -> Result<Archive<Box<dyn Read>>> {
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

fn extract_entry_to_file<R: Read>(entry: &mut tar::Entry<'_, R>, dest: &Path) -> Result<()> {
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

    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tar::Builder;
    use uuid::Uuid;

    #[test]
    fn ensure_sortformer_artifacts_extracts_required_files_from_gzipped_nemo() {
        let temp_dir = std::env::temp_dir().join(format!("sortformer-nemo-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let model_dir = temp_dir.join(ModelVariant::DiarStreamingSortformer4SpkV21.dir_name());
        fs::create_dir_all(&model_dir).unwrap();

        let nemo_path = model_dir.join("diar_streaming_sortformer_4spk-v2.1.nemo");
        let nemo_file = File::create(&nemo_path).unwrap();
        let encoder = GzEncoder::new(nemo_file, Compression::default());
        let mut builder = Builder::new(encoder);

        add_archive_file(
            &mut builder,
            "nested/model_config.yaml",
            b"sample_rate: 16000\nmax_num_of_spks: 4\n",
        );
        add_archive_file(&mut builder, "nested/model_weights.ckpt", b"checkpoint");
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap();

        let artifacts =
            ensure_sortformer_artifacts(&model_dir, ModelVariant::DiarStreamingSortformer4SpkV21)
                .unwrap();

        assert_eq!(
            fs::read_to_string(artifacts.model_config_path).unwrap(),
            "sample_rate: 16000\nmax_num_of_spks: 4\n"
        );
        assert_eq!(fs::read(artifacts.checkpoint_path).unwrap(), b"checkpoint");

        fs::remove_dir_all(temp_dir).unwrap();
    }

    /// A plain-tar `.nemo` holding `config` and `checkpoint`, modified at
    /// `modified_secs` past the epoch.
    fn write_nemo(path: &Path, config: &[u8], checkpoint: &[u8], modified_secs: u64) {
        let mut builder = Builder::new(File::create(path).unwrap());
        add_archive_file(&mut builder, "./model_config.yaml", config);
        add_archive_file(&mut builder, "./model_weights.ckpt", checkpoint);
        builder.into_inner().unwrap().sync_all().unwrap();
        set_modified(path, modified_secs);
    }

    fn set_modified(path: &Path, modified_secs: u64) {
        File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(UNIX_EPOCH + std::time::Duration::from_secs(modified_secs))
            .unwrap();
    }

    /// NDZ-5: the extraction is tied to the archive it came from. A
    /// re-published archive (new size, or same size and new bytes) is
    /// extracted again; an unchanged or merely touched one is not.
    #[test]
    fn a_changed_nemo_archive_refreshes_the_extraction() {
        let temp_dir = std::env::temp_dir().join(format!("sortformer-nemo-{}", Uuid::new_v4()));
        let variant = ModelVariant::Nemotron3Diarization;
        let model_dir = temp_dir.join(variant.dir_name());
        fs::create_dir_all(&model_dir).unwrap();
        let nemo_path = model_dir.join("Nemotron-3-Diarization.nemo");
        let config = |artifacts: &SortformerArtifacts| {
            fs::read_to_string(&artifacts.model_config_path).unwrap()
        };

        write_nemo(&nemo_path, b"revision: 1\n", b"weights-1", 1_000);
        let artifacts = ensure_sortformer_artifacts(&model_dir, variant).unwrap();
        assert_eq!(config(&artifacts), "revision: 1\n");

        // Unchanged: the extracted files are served as they are.
        fs::write(&artifacts.model_config_path, "kept\n").unwrap();
        let artifacts = ensure_sortformer_artifacts(&model_dir, variant).unwrap();
        assert_eq!(config(&artifacts), "kept\n");

        // Touched (same bytes, new time): the recorded digest still matches.
        set_modified(&nemo_path, 2_000);
        let artifacts = ensure_sortformer_artifacts(&model_dir, variant).unwrap();
        assert_eq!(config(&artifacts), "kept\n");

        // Re-published with the same size and new bytes.
        write_nemo(&nemo_path, b"revision: 2\n", b"weights-2", 3_000);
        let artifacts = ensure_sortformer_artifacts(&model_dir, variant).unwrap();
        assert_eq!(config(&artifacts), "revision: 2\n");
        assert_eq!(fs::read(&artifacts.checkpoint_path).unwrap(), b"weights-2");

        // Re-published with a new size (the same time stamp).
        let larger = vec![3u8; 64 * 1024];
        write_nemo(&nemo_path, b"revision: 3\n", &larger, 3_000);
        let artifacts = ensure_sortformer_artifacts(&model_dir, variant).unwrap();
        assert_eq!(config(&artifacts), "revision: 3\n");
        assert_eq!(fs::read(&artifacts.checkpoint_path).unwrap(), larger);

        // An extraction from before stamps is refreshed once.
        fs::remove_file(artifacts.extracted_dir.join(EXTRACTION_STAMP_FILE)).unwrap();
        fs::write(&artifacts.model_config_path, "stale\n").unwrap();
        let artifacts = ensure_sortformer_artifacts(&model_dir, variant).unwrap();
        assert_eq!(config(&artifacts), "revision: 3\n");

        fs::remove_dir_all(temp_dir).unwrap();
    }

    fn add_archive_file<W: Write>(builder: &mut Builder<W>, path: &str, contents: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, contents).unwrap();
    }
}

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_core::pickle::read_pth_tensor_info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::weights::nemo::{
    ensure_free_disk, extract_entry_to_file, open_nemo_archive,
};

#[derive(Debug, Clone)]
pub struct ParakeetArtifacts {
    pub nemo_path: PathBuf,
    pub extracted_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub model_config_path: PathBuf,
    pub tokenizer_vocab_path: PathBuf,
}

/// Where `ensure_parakeet_artifacts` finds the `.nemo` archive and puts the
/// checkpoint it extracts, computed without touching the filesystem. Load
/// admission reads these headers (`runtime/lifecycle/asr_memory.rs`).
#[derive(Debug, Clone)]
pub(crate) struct ParakeetCheckpointPaths {
    pub(crate) nemo_path: PathBuf,
    pub(crate) extracted_dir: PathBuf,
    pub(crate) checkpoint_path: PathBuf,
    /// The zip-repacked copy loaded when Candle cannot read the original.
    pub(crate) repacked_checkpoint_path: PathBuf,
}

pub(crate) fn parakeet_checkpoint_paths(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<ParakeetCheckpointPaths> {
    let nemo_filename = match variant {
        ModelVariant::ParakeetTdt06BV3 => "parakeet-tdt-0.6b-v3.nemo",
        _ => {
            return Err(Error::InvalidInput(format!(
                "Unsupported Parakeet variant: {}",
                variant.dir_name()
            )));
        }
    };
    let extracted_dir = model_dir.join("parakeet-native");
    Ok(ParakeetCheckpointPaths {
        nemo_path: model_dir.join(nemo_filename),
        checkpoint_path: extracted_dir.join("model_weights.ckpt"),
        repacked_checkpoint_path: extracted_dir.join("model_weights.repacked.ckpt"),
        extracted_dir,
    })
}

/// Whether `parakeet-native/` holds everything a load reads (config,
/// tokenizer and a checkpoint, original or repacked), so the model is ready
/// without its `.nemo`.
pub(crate) fn parakeet_native_artifacts_complete(model_dir: &Path, variant: ModelVariant) -> bool {
    parakeet_checkpoint_paths(model_dir, variant).is_ok_and(|paths| {
        paths.extracted_dir.join("model_config.yaml").is_file()
            && paths.extracted_dir.join("tokenizer.vocab").is_file()
            && (paths.checkpoint_path.is_file() || paths.repacked_checkpoint_path.is_file())
    })
}

pub fn ensure_parakeet_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<ParakeetArtifacts> {
    let ParakeetCheckpointPaths {
        nemo_path,
        extracted_dir,
        checkpoint_path,
        repacked_checkpoint_path,
    } = parakeet_checkpoint_paths(model_dir, variant)?;
    // A complete extracted directory is enough on its own: the `.nemo` is
    // only its source and may be deleted to reclaim its 2.5 GB (PKT-9).
    if !nemo_path.exists() && !parakeet_native_artifacts_complete(model_dir, variant) {
        return Err(Error::ModelNotFound(format!(
            "Missing .nemo checkpoint for {} at {}",
            variant.dir_name(),
            nemo_path.display()
        )));
    }

    fs::create_dir_all(&extracted_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create Parakeet cache directory {}: {}",
            extracted_dir.display(),
            e
        ))
    })?;

    let model_config_path = extracted_dir.join("model_config.yaml");
    let tokenizer_vocab_path = extracted_dir.join("tokenizer.vocab");

    let has_checkpoint = checkpoint_path.exists() || repacked_checkpoint_path.exists();
    if !(model_config_path.exists() && tokenizer_vocab_path.exists() && has_checkpoint) {
        extract_from_nemo(
            &nemo_path,
            &model_config_path,
            &tokenizer_vocab_path,
            &checkpoint_path,
        )?;
    }

    let checkpoint_path = normalize_checkpoint_if_needed(
        &checkpoint_path,
        &extracted_dir,
        &repacked_checkpoint_path,
    )?;

    Ok(ParakeetArtifacts {
        nemo_path,
        extracted_dir,
        checkpoint_path,
        model_config_path,
        tokenizer_vocab_path,
    })
}

/// Stream the three members a load reads out of the archive in one pass:
/// each goes straight to its file through a small copy buffer, after a free
/// disk check, instead of `tar -xOf` collecting the 2.5 GB checkpoint in host
/// memory first (PKT-9).
fn extract_from_nemo(
    nemo_path: &Path,
    model_config_path: &Path,
    tokenizer_vocab_path: &Path,
    checkpoint_path: &Path,
) -> Result<()> {
    let mut archive = open_nemo_archive(nemo_path)?;
    let mut members = [
        ("model_config.yaml", model_config_path, false),
        ("tokenizer.vocab", tokenizer_vocab_path, false),
        ("model_weights.ckpt", checkpoint_path, false),
    ];
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
        if let Some(member) = members
            .iter_mut()
            .find(|(suffix, _, found)| !*found && entry_path.ends_with(suffix))
        {
            extract_entry_to_file(&mut entry, member.1)?;
            member.2 = true;
        }
    }
    if let Some((suffix, _, _)) = members.iter().find(|(_, _, found)| !*found) {
        return Err(Error::ModelLoadError(format!(
            "Unable to locate {suffix} inside .nemo archive {}",
            nemo_path.display()
        )));
    }
    Ok(())
}

fn normalize_checkpoint_if_needed(
    checkpoint_path: &Path,
    extracted_dir: &Path,
    repacked: &Path,
) -> Result<PathBuf> {
    if read_pth_tensor_info(checkpoint_path, false, None).is_ok() {
        return Ok(checkpoint_path.to_path_buf());
    }

    if repacked.exists() && read_pth_tensor_info(repacked, false, None).is_ok() {
        return Ok(repacked.to_path_buf());
    }

    // Unpacking and repacking hold two more copies of the checkpoint on disk.
    let checkpoint_bytes = fs::metadata(checkpoint_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    ensure_free_disk(
        extracted_dir,
        checkpoint_bytes.saturating_mul(2),
        &format!("repacking {}", checkpoint_path.display()),
    )?;
    let temp_extract_dir = extracted_dir.join("checkpoint_repack_tmp");
    if temp_extract_dir.exists() {
        let _ = fs::remove_dir_all(&temp_extract_dir);
    }
    fs::create_dir_all(&temp_extract_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create repack temp directory {}: {}",
            temp_extract_dir.display(),
            e
        ))
    })?;

    let unzip_status = Command::new("unzip")
        .arg("-qq")
        .arg("-o")
        .arg(checkpoint_path)
        .arg("-d")
        .arg(&temp_extract_dir)
        .status()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to run unzip on {}: {}",
                checkpoint_path.display(),
                e
            ))
        })?;

    if !unzip_status.success() {
        return Err(Error::ModelLoadError(format!(
            "Failed to unpack checkpoint {} for normalization",
            checkpoint_path.display()
        )));
    }

    let repacked_tmp = extracted_dir.join("model_weights.repacked.tmp.ckpt");
    let zip_status = Command::new("zip")
        .arg("-q")
        .arg("-0")
        .arg("-r")
        .arg(&repacked_tmp)
        .arg("model_weights")
        .current_dir(&temp_extract_dir)
        .status()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to run zip while normalizing {}: {}",
                checkpoint_path.display(),
                e
            ))
        })?;

    let _ = fs::remove_dir_all(&temp_extract_dir);

    if !zip_status.success() {
        return Err(Error::ModelLoadError(format!(
            "Failed to repack checkpoint {} into a readable format",
            checkpoint_path.display()
        )));
    }

    fs::rename(&repacked_tmp, repacked).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to finalize repacked checkpoint {}: {}",
            repacked.display(),
            e
        ))
    })?;

    // Validate once before returning.
    read_pth_tensor_info(repacked, false, None).map_err(|e| {
        Error::ModelLoadError(format!(
            "Repacked checkpoint is still unreadable ({}): {}",
            repacked.display(),
            e
        ))
    })?;

    Ok(repacked.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tar::Builder;
    use uuid::Uuid;

    fn add_archive_file(builder: &mut Builder<File>, path: &str, contents: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, contents).unwrap();
    }

    /// The members stream out of the archive in one pass into their files;
    /// a missing member is reported by name.
    #[test]
    fn extraction_streams_each_member_to_its_file() {
        let dir = std::env::temp_dir().join(format!("parakeet-nemo-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let nemo = dir.join("model.nemo");
        let mut builder = Builder::new(File::create(&nemo).unwrap());
        add_archive_file(&mut builder, "./model_config.yaml", b"config");
        add_archive_file(&mut builder, "./abc_tokenizer.vocab", b"vocab");
        add_archive_file(&mut builder, "./model_weights.ckpt", &[7u8; 4096]);
        builder.finish().unwrap();
        drop(builder);

        let (config, vocab, ckpt) = (
            dir.join("model_config.yaml"),
            dir.join("tokenizer.vocab"),
            dir.join("model_weights.ckpt"),
        );
        extract_from_nemo(&nemo, &config, &vocab, &ckpt).unwrap();
        assert_eq!(fs::read(&config).unwrap(), b"config");
        assert_eq!(fs::read(&vocab).unwrap(), b"vocab");
        assert_eq!(fs::read(&ckpt).unwrap(), vec![7u8; 4096]);

        let partial = dir.join("partial.nemo");
        let mut builder = Builder::new(File::create(&partial).unwrap());
        add_archive_file(&mut builder, "./model_config.yaml", b"config");
        builder.finish().unwrap();
        drop(builder);
        let error = extract_from_nemo(&partial, &config, &vocab, &ckpt)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tokenizer.vocab"), "{error}");
        fs::remove_dir_all(dir).unwrap();
    }

    /// A complete `parakeet-native/` (original or repacked checkpoint) is
    /// ready without the `.nemo`.
    #[test]
    fn complete_native_directory_needs_no_nemo() {
        let variant = ModelVariant::ParakeetTdt06BV3;
        let dir = std::env::temp_dir().join(format!("parakeet-native-{}", Uuid::new_v4()));
        let paths = parakeet_checkpoint_paths(&dir, variant).unwrap();
        fs::create_dir_all(&paths.extracted_dir).unwrap();
        fs::write(paths.extracted_dir.join("model_config.yaml"), "config").unwrap();
        fs::write(paths.extracted_dir.join("tokenizer.vocab"), "vocab").unwrap();
        assert!(!parakeet_native_artifacts_complete(&dir, variant));
        assert!(matches!(
            ensure_parakeet_artifacts(&dir, variant),
            Err(Error::ModelNotFound(_))
        ));
        fs::write(&paths.repacked_checkpoint_path, "ckpt").unwrap();
        assert!(parakeet_native_artifacts_complete(&dir, variant));
        fs::remove_file(&paths.repacked_checkpoint_path).unwrap();
        fs::write(&paths.checkpoint_path, "ckpt").unwrap();
        assert!(parakeet_native_artifacts_complete(&dir, variant));
        fs::remove_dir_all(dir).unwrap();
    }
}

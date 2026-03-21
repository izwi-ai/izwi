use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_core::pickle::read_pth_tensor_info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

#[derive(Debug, Clone)]
pub struct ParakeetArtifacts {
    pub nemo_path: PathBuf,
    pub extracted_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub model_config_path: PathBuf,
    pub tokenizer_vocab_path: PathBuf,
}

pub fn ensure_parakeet_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<ParakeetArtifacts> {
    let nemo_filename = match variant {
        ModelVariant::ParakeetTdt06BV3 => "parakeet-tdt-0.6b-v3.nemo",
        _ => {
            return Err(Error::InvalidInput(format!(
                "Unsupported Parakeet variant: {}",
                variant.dir_name()
            )));
        }
    };

    let nemo_path = model_dir.join(nemo_filename);
    if !nemo_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing .nemo checkpoint for {} at {}",
            variant.dir_name(),
            nemo_path.display()
        )));
    }

    let extracted_dir = model_dir.join("parakeet-native");
    fs::create_dir_all(&extracted_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create Parakeet cache directory {}: {}",
            extracted_dir.display(),
            e
        ))
    })?;

    let model_config_path = extracted_dir.join("model_config.yaml");
    let tokenizer_vocab_path = extracted_dir.join("tokenizer.vocab");
    let checkpoint_path = extracted_dir.join("model_weights.ckpt");

    if !(model_config_path.exists() && tokenizer_vocab_path.exists() && checkpoint_path.exists()) {
        extract_from_nemo(
            &nemo_path,
            &model_config_path,
            &tokenizer_vocab_path,
            &checkpoint_path,
        )?;
    }

    let checkpoint_path = normalize_checkpoint_if_needed(&checkpoint_path, &extracted_dir)?;

    Ok(ParakeetArtifacts {
        nemo_path,
        extracted_dir,
        checkpoint_path,
        model_config_path,
        tokenizer_vocab_path,
    })
}

fn extract_from_nemo(
    nemo_path: &Path,
    model_config_path: &Path,
    tokenizer_vocab_path: &Path,
    checkpoint_path: &Path,
) -> Result<()> {
    let listing = tar_list(nemo_path)?;

    let model_config_entry = find_tar_entry(&listing, "model_config.yaml")?;
    let tokenizer_vocab_entry = find_tar_entry(&listing, "tokenizer.vocab")?;
    let checkpoint_entry = find_tar_entry(&listing, "model_weights.ckpt")?;

    extract_tar_entry_to_file(nemo_path, &model_config_entry, model_config_path)?;
    extract_tar_entry_to_file(nemo_path, &tokenizer_vocab_entry, tokenizer_vocab_path)?;
    extract_tar_entry_to_file(nemo_path, &checkpoint_entry, checkpoint_path)?;

    Ok(())
}

fn tar_list(nemo_path: &Path) -> Result<Vec<String>> {
    let output = Command::new("tar")
        .arg("-tf")
        .arg(nemo_path)
        .output()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to list .nemo archive {}: {}",
                nemo_path.display(),
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ModelLoadError(format!(
            "Failed to list .nemo archive {}: {}",
            nemo_path.display(),
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn find_tar_entry(entries: &[String], suffix: &str) -> Result<String> {
    entries
        .iter()
        .find(|e| e.ends_with(suffix))
        .cloned()
        .ok_or_else(|| {
            Error::ModelLoadError(format!("Unable to locate {} inside .nemo archive", suffix))
        })
}

fn extract_tar_entry_to_file(nemo_path: &Path, entry: &str, dest: &Path) -> Result<()> {
    let tmp_path = dest.with_extension("tmp");
    let mut tmp_file = File::create(&tmp_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed creating temp extraction file {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    let output = Command::new("tar")
        .arg("-xOf")
        .arg(nemo_path)
        .arg(entry)
        .output()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed extracting {} from {}: {}",
                entry,
                nemo_path.display(),
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ModelLoadError(format!(
            "Failed extracting {} from {}: {}",
            entry,
            nemo_path.display(),
            stderr.trim()
        )));
    }

    tmp_file.write_all(&output.stdout).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed writing extracted file {}: {}",
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

fn normalize_checkpoint_if_needed(checkpoint_path: &Path, extracted_dir: &Path) -> Result<PathBuf> {
    if read_pth_tensor_info(checkpoint_path, false, None).is_ok() {
        return Ok(checkpoint_path.to_path_buf());
    }

    let repacked = extracted_dir.join("model_weights.repacked.ckpt");
    if repacked.exists() && read_pth_tensor_info(&repacked, false, None).is_ok() {
        return Ok(repacked);
    }

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

    fs::rename(&repacked_tmp, &repacked).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to finalize repacked checkpoint {}: {}",
            repacked.display(),
            e
        ))
    })?;

    // Validate once before returning.
    read_pth_tensor_info(&repacked, false, None).map_err(|e| {
        Error::ModelLoadError(format!(
            "Repacked checkpoint is still unreadable ({}): {}",
            repacked.display(),
            e
        ))
    })?;

    Ok(repacked)
}

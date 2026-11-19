//! Audio8 artifact validation.
//!
//! The published checkpoint is one `model.safetensors` (slow and fast
//! transformers, flat tensor names) beside `codec.pth`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

pub const AUDIO8_WEIGHTS_FILE: &str = "model.safetensors";
pub const AUDIO8_CODEC_FILE: &str = "codec.pth";

pub const AUDIO8_REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    AUDIO8_WEIGHTS_FILE,
    AUDIO8_CODEC_FILE,
];

#[derive(Debug, Clone)]
pub struct Audio8ArtifactManifest {
    pub model_dir: PathBuf,
    pub shard_files: Vec<String>,
    pub tensor_count: usize,
    pub slow_tensor_count: usize,
    pub fast_tensor_count: usize,
    pub codec_path: PathBuf,
}

impl Audio8ArtifactManifest {
    pub fn load(model_dir: &Path) -> Result<Self> {
        validate_required_files(model_dir)?;
        let names = safetensors_tensor_names(&model_dir.join(AUDIO8_WEIGHTS_FILE))?;
        let slow_tensor_count = names
            .iter()
            .filter(|name| name.starts_with("layers."))
            .count();
        let fast_tensor_count = names
            .iter()
            .filter(|name| name.starts_with("fast_layers."))
            .count();
        if slow_tensor_count == 0 {
            return Err(Error::ModelLoadError(
                "Audio8 checkpoint contains no slow transformer tensors".to_string(),
            ));
        }
        if fast_tensor_count == 0 {
            return Err(Error::ModelLoadError(
                "Audio8 checkpoint contains no fast transformer tensors".to_string(),
            ));
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            shard_files: vec![AUDIO8_WEIGHTS_FILE.to_string()],
            tensor_count: names.len(),
            slow_tensor_count,
            fast_tensor_count,
            codec_path: model_dir.join(AUDIO8_CODEC_FILE),
        })
    }

    /// Stable content identity for durable speech recovery, computed once per
    /// native load with bounded scratch space (never once per speech segment).
    pub fn content_fingerprint(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let files: BTreeSet<&str> = AUDIO8_REQUIRED_FILES
            .iter()
            .copied()
            .chain(self.shard_files.iter().map(String::as_str))
            .collect();
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        for name in files {
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            let path = self.model_dir.join(name);
            let mut file =
                fs::File::open(&path).map_err(|error| Error::ModelLoadError(error.to_string()))?;
            digest.update(
                file.metadata()
                    .map_err(|error| Error::ModelLoadError(error.to_string()))?
                    .len()
                    .to_le_bytes(),
            );
            loop {
                let length = file
                    .read(&mut buffer)
                    .map_err(|error| Error::ModelLoadError(error.to_string()))?;
                if length == 0 {
                    break;
                }
                digest.update(&buffer[..length]);
            }
        }
        Ok(hex::encode(digest.finalize()))
    }

    pub fn shard_paths(&self) -> Vec<PathBuf> {
        self.shard_files
            .iter()
            .map(|file| self.model_dir.join(file))
            .collect()
    }
}

fn validate_required_files(model_dir: &Path) -> Result<()> {
    for file in AUDIO8_REQUIRED_FILES {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Audio8 required artifact missing: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Tensor names from the safetensors header, without mapping the payload.
pub(crate) fn safetensors_tensor_names(path: &Path) -> Result<Vec<String>> {
    // SAFETY: the read-only mapping is dropped before this function returns
    // and only the validated header is read.
    let safe =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }.map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to parse Audio8 checkpoint {}: {err}",
                path.display()
            ))
        })?;
    Ok(safe.tensors().into_iter().map(|(name, _)| name).collect())
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use candle_core::{Device, Tensor};

    /// A model directory holding every required file; the checkpoint has one
    /// tiny tensor per name in `tensors`.
    pub(crate) fn write_artifacts(label: &str, tensors: &[&str]) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("izwi-audio8-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            std::fs::write(dir.join(file), "{}").unwrap();
        }
        std::fs::write(dir.join(super::AUDIO8_CODEC_FILE), b"codec").unwrap();
        write_checkpoint(&dir, tensors);
        dir
    }

    pub(crate) fn write_checkpoint(dir: &Path, tensors: &[&str]) {
        let tensors = tensors
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    Tensor::zeros((1, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        candle_core::safetensors::save(&tensors, dir.join(super::AUDIO8_WEIGHTS_FILE)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::write_artifacts;
    use super::*;

    #[test]
    fn manifest_counts_slow_and_fast_tensors() {
        let dir = write_artifacts(
            "artifacts",
            &[
                "embeddings.weight",
                "layers.0.attention.wqkv.weight",
                "layers.0.attention.wqkv.bias",
                "fast_layers.0.attention.wqkv.weight",
            ],
        );
        let manifest = Audio8ArtifactManifest::load(&dir).unwrap();
        assert_eq!(manifest.shard_files, vec![AUDIO8_WEIGHTS_FILE]);
        assert_eq!(manifest.tensor_count, 4);
        assert_eq!(manifest.slow_tensor_count, 2);
        assert_eq!(manifest.fast_tensor_count, 1);
        assert_eq!(manifest.codec_path, dir.join(AUDIO8_CODEC_FILE));
        assert_eq!(manifest.content_fingerprint().unwrap().len(), 64);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn manifest_requires_every_file_and_both_transformers() {
        let dir = write_artifacts("artifacts-missing", &["layers.0.attention.wqkv.weight"]);
        let err = Audio8ArtifactManifest::load(&dir).unwrap_err();
        assert!(err.to_string().contains("fast transformer"));
        std::fs::remove_file(dir.join(AUDIO8_CODEC_FILE)).unwrap();
        let err = Audio8ArtifactManifest::load(&dir).unwrap_err();
        assert!(err.to_string().contains("codec.pth"));
        let _ = std::fs::remove_dir_all(dir);
    }
}

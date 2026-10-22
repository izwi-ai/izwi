//! Fish S2 artifact validation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

pub const FISH_S2_REQUIRED_FILES: &[&str] = &[
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "codec.pth",
];

#[derive(Debug, Clone)]
pub struct FishS2ArtifactManifest {
    pub model_dir: PathBuf,
    pub shard_files: Vec<String>,
    pub tensor_count: usize,
    pub text_tensor_count: usize,
    pub audio_decoder_tensor_count: usize,
    pub codec_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

impl FishS2ArtifactManifest {
    pub fn load(model_dir: &Path) -> Result<Self> {
        validate_required_files(model_dir)?;
        let index = read_safetensors_index(model_dir)?;
        let mut shard_files = BTreeSet::new();
        let mut text_tensor_count = 0usize;
        let mut audio_decoder_tensor_count = 0usize;

        for (tensor_name, shard_name) in &index.weight_map {
            validate_shard_name(shard_name)?;
            shard_files.insert(shard_name.clone());
            if tensor_name.starts_with("text_model.") {
                text_tensor_count += 1;
            }
            if tensor_name.starts_with("audio_decoder.") {
                audio_decoder_tensor_count += 1;
            }
        }

        if shard_files.is_empty() {
            return Err(Error::ModelLoadError(
                "Fish S2 safetensors index contains no shard files".to_string(),
            ));
        }
        if text_tensor_count == 0 {
            return Err(Error::ModelLoadError(
                "Fish S2 safetensors index contains no text_model tensors".to_string(),
            ));
        }
        if audio_decoder_tensor_count == 0 {
            return Err(Error::ModelLoadError(
                "Fish S2 safetensors index contains no audio_decoder tensors".to_string(),
            ));
        }

        let shard_files = shard_files.into_iter().collect::<Vec<_>>();
        for shard_name in &shard_files {
            let path = model_dir.join(shard_name);
            if !path.exists() {
                return Err(Error::ModelLoadError(format!(
                    "Fish S2 missing safetensors shard {}",
                    path.display()
                )));
            }
        }

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            shard_files,
            tensor_count: index.weight_map.len(),
            text_tensor_count,
            audio_decoder_tensor_count,
            codec_path: model_dir.join("codec.pth"),
        })
    }

    /// Stable content identity for durable speech recovery, computed once per
    /// native load with bounded scratch space (never once per speech segment).
    pub fn content_fingerprint(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let files: BTreeSet<&str> = FISH_S2_REQUIRED_FILES
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
    for file in FISH_S2_REQUIRED_FILES {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 required artifact missing: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn read_safetensors_index(model_dir: &Path) -> Result<SafetensorsIndex> {
    let path = model_dir.join("model.safetensors.index.json");
    let raw = fs::read_to_string(&path).map_err(|err| {
        Error::ModelLoadError(format!("Failed to read {}: {err}", path.display()))
    })?;
    serde_json::from_str(&raw)
        .map_err(|err| Error::ModelLoadError(format!("Failed to parse {}: {err}", path.display())))
}

fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path.components().count() != 1 || name.contains("..") || name.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Fish S2 safetensors index contains unsafe shard path `{name}`"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_model_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("izwi-fish-s2-artifacts-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_required(dir: &Path) {
        for file in FISH_S2_REQUIRED_FILES {
            std::fs::write(dir.join(file), "{}").unwrap();
        }
        std::fs::write(dir.join("chat_template.jinja"), "<|im_start|>{{ role }}").unwrap();
        std::fs::write(dir.join("codec.pth"), [0u8]).unwrap();
    }

    #[test]
    fn validates_fish_s2_artifact_manifest() {
        let dir = temp_model_dir();
        write_required(&dir);
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), [0u8]).unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), [0u8]).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{
                "text_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
                "audio_decoder.layers.0.self_attn.qkv_proj.weight":"model-00002-of-00002.safetensors"
            }}"#,
        )
        .unwrap();

        let manifest = FishS2ArtifactManifest::load(&dir).unwrap();
        assert_eq!(manifest.tensor_count, 2);
        assert_eq!(manifest.text_tensor_count, 1);
        assert_eq!(manifest.audio_decoder_tensor_count, 1);
        assert_eq!(manifest.shard_files.len(), 2);
        assert_eq!(manifest.shard_paths().len(), 2);
        let fingerprint = manifest.content_fingerprint().unwrap();
        assert_eq!(fingerprint, manifest.content_fingerprint().unwrap());
        std::fs::write(dir.join("tokenizer.json"), "changed tokenizer").unwrap();
        let tokenizer_changed = manifest.content_fingerprint().unwrap();
        assert_ne!(fingerprint, tokenizer_changed);
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), [1u8]).unwrap();
        assert_ne!(tokenizer_changed, manifest.content_fingerprint().unwrap());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_index_without_audio_decoder_tensors() {
        let dir = temp_model_dir();
        write_required(&dir);
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), [0u8]).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{
                "text_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors"
            }}"#,
        )
        .unwrap();

        let err = FishS2ArtifactManifest::load(&dir).unwrap_err();
        assert!(err.to_string().contains("audio_decoder"));

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_unsafe_shard_paths() {
        let dir = temp_model_dir();
        write_required(&dir);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{
                "text_model.model.embed_tokens.weight":"../escape.safetensors",
                "audio_decoder.layers.0.self_attn.qkv_proj.weight":"model-00002-of-00002.safetensors"
            }}"#,
        )
        .unwrap();

        let err = FishS2ArtifactManifest::load(&dir).unwrap_err();
        assert!(err.to_string().contains("unsafe shard path"));

        std::fs::remove_dir_all(dir).ok();
    }
}

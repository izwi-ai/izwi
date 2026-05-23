//! Preset voice catalog for Voxtral TTS.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::models::shared::weights::pytorch::read_single_tensor_pth;

use super::config::VoxtralTtsConfig;

pub const VOXTRAL_TTS_BUILT_IN_VOICES: [&str; 20] = [
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "it_male",
    "it_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "ar_male",
    "hi_male",
    "hi_female",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralVoiceInfo {
    pub name: String,
    pub id: usize,
    pub embedding_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralVoiceCatalog {
    voices: BTreeMap<String, VoxtralVoiceInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralVoiceEmbeddingShape {
    pub rank: usize,
    pub batch_size: Option<usize>,
    pub frames: usize,
    pub hidden_size: usize,
}

#[derive(Debug)]
pub struct VoxtralVoiceEmbeddingLibrary {
    catalog: VoxtralVoiceCatalog,
    device: Device,
    dtype: DType,
    expected_hidden_size: usize,
    cache: RwLock<BTreeMap<String, Tensor>>,
}

impl VoxtralVoiceCatalog {
    pub fn from_config(model_dir: &Path, config: &VoxtralTtsConfig) -> Result<Self> {
        let mut voices = BTreeMap::new();
        for (name, id) in &config.multimodal.audio_tokenizer_args.voice {
            let embedding_path = voice_embedding_path(model_dir, name);
            voices.insert(
                name.clone(),
                VoxtralVoiceInfo {
                    name: name.clone(),
                    id: *id,
                    embedding_path,
                },
            );
        }
        if voices.is_empty() {
            return Err(Error::ConfigError(
                "Voxtral TTS voice catalog is empty".to_string(),
            ));
        }
        Ok(Self { voices })
    }

    pub fn len(&self) -> usize {
        self.voices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }

    pub fn contains(&self, voice: &str) -> bool {
        self.voices.contains_key(voice)
    }

    pub fn resolve(&self, voice: &str) -> Result<&VoxtralVoiceInfo> {
        self.voices.get(voice).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unsupported Voxtral TTS voice `{voice}`. Available voices: {}",
                self.names_by_id().join(", ")
            ))
        })
    }

    pub fn names_by_id(&self) -> Vec<String> {
        let mut entries = self.voices.values().collect::<Vec<_>>();
        entries.sort_by_key(|voice| voice.id);
        entries
            .into_iter()
            .map(|voice| voice.name.clone())
            .collect()
    }

    pub fn iter_by_id(&self) -> Vec<&VoxtralVoiceInfo> {
        let mut entries = self.voices.values().collect::<Vec<_>>();
        entries.sort_by_key(|voice| voice.id);
        entries
    }

    pub fn missing_embedding_paths(&self) -> Vec<PathBuf> {
        self.iter_by_id()
            .into_iter()
            .filter(|voice| !voice.embedding_path.exists())
            .map(|voice| voice.embedding_path.clone())
            .collect()
    }

    pub fn validate_embedding_files(&self) -> Result<()> {
        let missing = self.missing_embedding_paths();
        if missing.is_empty() {
            return Ok(());
        }
        let rendered = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::ModelLoadError(format!(
            "Voxtral TTS is missing preset voice embedding files: {rendered}"
        )))
    }
}

impl VoxtralVoiceEmbeddingLibrary {
    pub fn new(
        catalog: VoxtralVoiceCatalog,
        device: Device,
        dtype: DType,
        expected_hidden_size: usize,
    ) -> Self {
        Self {
            catalog,
            device,
            dtype,
            expected_hidden_size,
            cache: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn available_speakers(&self) -> Vec<String> {
        self.catalog.names_by_id()
    }

    pub fn load(&self, voice: &str) -> Result<Tensor> {
        if let Some(cached) = self
            .cache
            .read()
            .map_err(|_| Error::ModelLoadError("Voxtral voice cache lock poisoned".to_string()))?
            .get(voice)
            .cloned()
        {
            return Ok(cached);
        }

        let info = self.catalog.resolve(voice)?;
        let tensor =
            read_single_tensor_pth(&info.embedding_path, &self.device, self.dtype, &info.name)?;
        let tensor = normalize_voice_embedding_tensor(
            tensor,
            self.expected_hidden_size,
            &info.embedding_path,
        )?;

        self.cache
            .write()
            .map_err(|_| Error::ModelLoadError("Voxtral voice cache lock poisoned".to_string()))?
            .insert(voice.to_string(), tensor.clone());
        Ok(tensor)
    }
}

pub fn validate_voice_embedding_shape(
    dims: &[usize],
    expected_hidden_size: usize,
) -> Result<VoxtralVoiceEmbeddingShape> {
    if dims.len() != 2 && dims.len() != 3 {
        return Err(Error::ModelLoadError(format!(
            "Voxtral voice embedding rank must be 2 or 3, got shape {:?}",
            dims
        )));
    }
    if dims.iter().any(|dim| *dim == 0) {
        return Err(Error::ModelLoadError(format!(
            "Voxtral voice embedding dimensions must be non-zero, got shape {:?}",
            dims
        )));
    }
    let hidden_size = *dims.last().unwrap_or(&0);
    if hidden_size != expected_hidden_size {
        return Err(Error::ModelLoadError(format!(
            "Voxtral voice embedding hidden size {hidden_size} does not match text hidden size {expected_hidden_size}"
        )));
    }

    let (batch_size, frames) = if dims.len() == 2 {
        (None, dims[0])
    } else {
        (Some(dims[0]), dims[1])
    };

    Ok(VoxtralVoiceEmbeddingShape {
        rank: dims.len(),
        batch_size,
        frames,
        hidden_size,
    })
}

fn normalize_voice_embedding_tensor(
    tensor: Tensor,
    expected_hidden_size: usize,
    path: &Path,
) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    validate_voice_embedding_shape(&dims, expected_hidden_size)
        .map_err(|err| Error::ModelLoadError(format!("{} in {}", err, path.display())))?;
    if dims.len() == 2 {
        tensor.unsqueeze(0).map_err(Error::from)
    } else {
        Ok(tensor)
    }
}

pub fn voice_embedding_path(model_dir: &Path, voice: &str) -> PathBuf {
    model_dir
        .join("voice_embedding")
        .join(format!("{voice}.pt"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};

    #[test]
    fn builds_catalog_from_official_voice_map() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let catalog =
            VoxtralVoiceCatalog::from_config(Path::new("/models/voxtral"), &config).unwrap();
        assert_eq!(catalog.len(), 20);
        assert!(catalog.contains("casual_male"));
        assert_eq!(catalog.names_by_id()[0], "casual_female");
        assert_eq!(catalog.names_by_id()[19], "hi_female");
        let neutral = catalog.resolve("neutral_male").unwrap();
        assert_eq!(neutral.id, 4);
        assert_eq!(
            neutral.embedding_path,
            Path::new("/models/voxtral")
                .join("voice_embedding")
                .join("neutral_male.pt")
        );
    }

    #[test]
    fn built_in_voice_constant_matches_params_order() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        assert_eq!(config.voice_names_by_id(), VOXTRAL_TTS_BUILT_IN_VOICES);
    }

    #[test]
    fn validates_voice_embedding_shapes_against_text_hidden_size() {
        let shape = validate_voice_embedding_shape(&[12, 3072], 3072).unwrap();
        assert_eq!(shape.rank, 2);
        assert_eq!(shape.batch_size, None);
        assert_eq!(shape.frames, 12);
        assert_eq!(shape.hidden_size, 3072);

        let batched = validate_voice_embedding_shape(&[1, 12, 3072], 3072).unwrap();
        assert_eq!(batched.rank, 3);
        assert_eq!(batched.batch_size, Some(1));
        assert_eq!(batched.frames, 12);

        assert!(validate_voice_embedding_shape(&[12, 2048], 3072).is_err());
        assert!(validate_voice_embedding_shape(&[3072], 3072).is_err());
        assert!(validate_voice_embedding_shape(&[1, 0, 3072], 3072).is_err());
    }
}

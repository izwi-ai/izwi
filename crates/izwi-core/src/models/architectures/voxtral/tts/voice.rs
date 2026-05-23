//! Preset voice catalog for Voxtral TTS.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

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
}

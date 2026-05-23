//! High-level Voxtral TTS model contract.

use std::path::{Path, PathBuf};

use candle_core::DType;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};

use super::codec::VoxtralCodecConfig;
use super::config::VoxtralTtsConfig;
use super::sampling::VoxtralTtsGenerationParams;
use super::voice::{voice_embedding_path, VoxtralVoiceCatalog};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxtralTtsDTypePlan {
    pub language_model: DType,
    pub acoustic_transformer: DType,
    pub codec: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralTtsAssets {
    pub params_path: PathBuf,
    pub tekken_path: PathBuf,
    pub weights_path: PathBuf,
    pub voice_embedding_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct VoxtralTtsOutput {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub frames_generated: usize,
}

pub struct VoxtralTtsModel {
    pub model_dir: PathBuf,
    pub config: VoxtralTtsConfig,
    pub voices: VoxtralVoiceCatalog,
    pub codec_config: VoxtralCodecConfig,
    pub dtype_plan: VoxtralTtsDTypePlan,
}

impl VoxtralTtsAssets {
    pub fn from_config(model_dir: &Path, config: &VoxtralTtsConfig) -> Self {
        Self {
            params_path: model_dir.join("params.json"),
            tekken_path: model_dir.join("tekken.json"),
            weights_path: model_dir.join("consolidated.safetensors"),
            voice_embedding_paths: config
                .voice_names_by_id()
                .iter()
                .map(|voice| voice_embedding_path(model_dir, voice))
                .collect(),
        }
    }

    pub fn missing_paths(&self) -> Vec<PathBuf> {
        let mut missing = Vec::new();
        for path in [&self.params_path, &self.tekken_path, &self.weights_path] {
            if !path.exists() {
                missing.push(path.clone());
            }
        }
        missing.extend(
            self.voice_embedding_paths
                .iter()
                .filter(|path| !path.exists())
                .cloned(),
        );
        missing
    }

    pub fn validate_present(&self) -> Result<()> {
        let missing = self.missing_paths();
        if missing.is_empty() {
            return Ok(());
        }
        let rendered = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::ModelLoadError(format!(
            "Voxtral TTS model directory is incomplete; missing {rendered}"
        )))
    }
}

impl VoxtralTtsModel {
    pub fn load_metadata(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        info!("Loading Voxtral TTS metadata from {:?}", model_dir);
        let config = VoxtralTtsConfig::load(model_dir)?;
        let assets = VoxtralTtsAssets::from_config(model_dir, &config);
        assets.validate_present()?;
        let voices = VoxtralVoiceCatalog::from_config(model_dir, &config)?;
        voices.validate_embedding_files()?;
        let codec_config = VoxtralCodecConfig::from_config(&config)?;
        let dtype_plan =
            select_voxtral_tts_dtypes(&device, voxtral_tts_dtype_override().as_deref())?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            config,
            voices,
            codec_config,
            dtype_plan,
        })
    }

    pub fn generate_with_voice(
        &self,
        _text: &str,
        _voice: &str,
        _params: VoxtralTtsGenerationParams,
    ) -> Result<VoxtralTtsOutput> {
        Err(Error::InferenceError(
            "Voxtral TTS generation is not wired until the acoustic transformer and codec phases are complete"
                .to_string(),
        ))
    }
}

pub fn select_voxtral_tts_dtypes(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
) -> Result<VoxtralTtsDTypePlan> {
    if let Some(raw) = dtype_override.map(str::trim).filter(|raw| !raw.is_empty()) {
        let dtype =
            device.select_model_dtype_checked(ModelFamily::VoxtralTts, Some(raw), "Voxtral TTS")?;
        return Ok(VoxtralTtsDTypePlan {
            language_model: dtype,
            acoustic_transformer: dtype,
            codec: dtype,
        });
    }

    let transformer_dtype = device.select_model_dtype(ModelFamily::VoxtralTts, None);
    let codec_dtype = if device.kind.is_cuda() {
        transformer_dtype
    } else {
        DType::F32
    };
    Ok(VoxtralTtsDTypePlan {
        language_model: transformer_dtype,
        acoustic_transformer: transformer_dtype,
        codec: codec_dtype,
    })
}

fn voxtral_tts_dtype_override() -> Option<String> {
    std::env::var("IZWI_VOXTRAL_TTS_DTYPE")
        .ok()
        .or_else(|| std::env::var("IZWI_VOXTRAL_DTYPE").ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    use crate::backends::{DeviceCapabilities, DeviceKind};
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};

    fn profile(kind: DeviceKind, supports_bf16: bool, supports_f16: bool) -> DeviceProfile {
        DeviceProfile {
            device: Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_bf16,
                supports_f16,
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn asset_contract_uses_hf_file_layout() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let assets =
            VoxtralTtsAssets::from_config(Path::new("/models/Voxtral-4B-TTS-2603"), &config);
        assert_eq!(
            assets.params_path,
            Path::new("/models/Voxtral-4B-TTS-2603").join("params.json")
        );
        assert_eq!(
            assets.weights_path,
            Path::new("/models/Voxtral-4B-TTS-2603").join("consolidated.safetensors")
        );
        assert_eq!(assets.voice_embedding_paths.len(), 20);
        assert_eq!(
            assets.voice_embedding_paths[1],
            Path::new("/models/Voxtral-4B-TTS-2603")
                .join("voice_embedding")
                .join("casual_male.pt")
        );
    }

    #[test]
    fn dtype_plan_keeps_cpu_codec_in_f32_and_allows_cuda_bf16() {
        let cpu = profile(DeviceKind::Cpu, false, false);
        let cpu_plan = select_voxtral_tts_dtypes(&cpu, None).unwrap();
        assert_eq!(cpu_plan.language_model, DType::F32);
        assert_eq!(cpu_plan.codec, DType::F32);

        let cuda = profile(DeviceKind::Cuda, true, true);
        let cuda_plan = select_voxtral_tts_dtypes(&cuda, None).unwrap();
        assert_eq!(cuda_plan.language_model, DType::BF16);
        assert_eq!(cuda_plan.acoustic_transformer, DType::BF16);
        assert_eq!(cuda_plan.codec, DType::BF16);
    }

    #[test]
    fn dtype_override_applies_to_all_voxtral_tts_stages() {
        let cuda = profile(DeviceKind::Cuda, true, true);
        let plan = select_voxtral_tts_dtypes(&cuda, Some("f16")).unwrap();
        assert_eq!(plan.language_model, DType::F16);
        assert_eq!(plan.acoustic_transformer, DType::F16);
        assert_eq!(plan.codec, DType::F16);
    }
}

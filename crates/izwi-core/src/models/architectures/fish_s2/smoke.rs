use std::env;
use std::path::PathBuf;

use crate::runtime::{GenerationRequest, RuntimeService};
use crate::{ContextLengthPreference, EngineConfig};
use base64::Engine as _;

use crate::backends::{BackendPreference, DeviceSelector};
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

use super::{FishS2DacConfig, FishS2TtsModel};

#[test]
#[ignore = "requires the pinned local Fish S2 bundle; reads metadata only"]
fn fish_s2_real_artifact_metadata_contracts() -> Result<()> {
    let dir = required_env_path("IZWI_FISH_S2_MODEL_DIR")?;
    let config = super::FishS2Config::load(&dir)?;
    let _tokenizer = super::FishS2PromptTokenizer::load(&dir, &config)?;
    let device = DeviceSelector::detect_for_preference(BackendPreference::Cpu)?;
    let memory = super::weights::fish_s2_model_memory(&dir, &device)?;
    assert!(memory.resident_bytes > 18 * 1024 * 1024 * 1024);
    assert!(memory.load_peak_bytes >= memory.resident_bytes);
    eprintln!("Pinned Fish CPU metadata: {memory:?}");
    Ok(())
}

#[test]
#[ignore = "requires local fishaudio/s2-pro artifacts"]
fn fish_s2_real_artifacts_load_native_modules() -> Result<()> {
    let model_dir = required_env_path("IZWI_FISH_S2_MODEL_DIR")?;
    let backend = env_backend()?;
    let device = DeviceSelector::detect_for_preference(backend)?;
    let model = FishS2TtsModel::load(&model_dir, ModelVariant::FishAudioS2Pro, device)?;

    assert!(model.runtime.is_some());
    assert_eq!(model.config.num_codebooks, 10);
    assert_eq!(model.config.codebook_size, 4096);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires local fishaudio/s2-pro artifacts and sufficient backend memory"]
async fn fish_s2_real_model_smoke_generates_finite_audio() -> Result<()> {
    let variant = ModelVariant::FishAudioS2Pro;
    let model_dir = required_env_path("IZWI_FISH_S2_MODEL_DIR")?;
    if model_dir.file_name().and_then(|name| name.to_str()) != Some(variant.dir_name()) {
        return Err(Error::InvalidInput(format!(
            "IZWI_FISH_S2_MODEL_DIR must be a runtime model directory named {}",
            variant.dir_name()
        )));
    }
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let reference_wav = env::var_os("IZWI_FISH_S2_REFERENCE_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture_dir.join("fox.wav"));
    let reference_text = match env::var("IZWI_FISH_S2_REFERENCE_TEXT") {
        Ok(text) => text,
        Err(_) if env::var_os("IZWI_FISH_S2_REFERENCE_WAV").is_some() => {
            return Err(Error::InvalidInput(
                "Set IZWI_FISH_S2_REFERENCE_TEXT to the exact transcript of the custom reference WAV".into()
            ));
        }
        Err(_) => std::fs::read_to_string(fixture_dir.join("fox.md"))?,
    };
    // Reproduce the checked-in reference first; a separate held-out target can
    // then measure synthesis quality without changing the conditioning fixture.
    let target_text =
        env::var("IZWI_FISH_S2_TARGET_TEXT").unwrap_or_else(|_| reference_text.clone());
    let max_frames = env_usize("IZWI_FISH_S2_SMOKE_MAX_FRAMES", 512)?;
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: model_dir
            .parent()
            .ok_or_else(|| Error::InvalidInput("Missing model parent".into()))?
            .to_path_buf(),
        backend: env_backend()?,
        max_sequence_length: ContextLengthPreference::explicit(env_usize(
            "IZWI_FISH_S2_SMOKE_CONTEXT",
            4096,
        )?)?,
        ..EngineConfig::default()
    })?;
    let mut request = GenerationRequest::new(target_text).with_model_variant(variant);
    request.config.streaming = false;
    request.config.options.max_tokens = max_frames;
    request.config.options.temperature = env_f32("IZWI_FISH_S2_SMOKE_TEMPERATURE", 0.0)?;
    request.config.options.top_p = env_f32("IZWI_FISH_S2_SMOKE_TOP_P", 1.0)?;
    request.reference_audio =
        Some(base64::engine::general_purpose::STANDARD.encode(std::fs::read(reference_wav)?));
    request.reference_text = Some(reference_text);
    runtime.load_model(variant).await?;
    let generated = runtime.generate(request).await;
    let unloaded = runtime.unload_model(variant).await;
    let output = generated?;
    unloaded?;

    let dac_config = FishS2DacConfig::current();
    assert_eq!(output.sample_rate, dac_config.sample_rate);
    assert!(output.samples.len() >= dac_config.samples_per_frame()?);
    assert!(output.samples.iter().all(|sample| sample.is_finite()));
    assert!(
        output.samples.iter().any(|sample| sample.abs() > 1e-5),
        "silent output"
    );
    assert!(
        output.duration_secs()
            <= max_frames as f32 * dac_config.samples_per_frame()? as f32
                / output.sample_rate as f32
                + 0.1
    );
    if let Some(path) = env::var_os("IZWI_FISH_S2_SMOKE_OUTPUT_WAV") {
        let mut writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: output.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .map_err(|err| Error::AudioError(err.to_string()))?;
        for sample in &output.samples {
            writer
                .write_sample(*sample)
                .map_err(|err| Error::AudioError(err.to_string()))?;
        }
        writer
            .finalize()
            .map_err(|err| Error::AudioError(err.to_string()))?;
    }
    eprintln!(
        "Fish S2 runtime smoke: {:.2}s audio, {:.3} RTF, diagnostics {:?}",
        output.duration_secs(),
        output.rtf(),
        output.diagnostics
    );
    Ok(())
}

fn required_env_path(name: &str) -> Result<PathBuf> {
    let raw = env::var(name).map_err(|_| {
        Error::InvalidInput(format!(
            "Set {name} to run the ignored Fish S2 real-model smoke test"
        ))
    })?;
    let path = PathBuf::from(raw);
    if !path.exists() {
        return Err(Error::InvalidInput(format!(
            "{name} path does not exist: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn env_backend() -> Result<BackendPreference> {
    let raw = env::var("IZWI_FISH_S2_BACKEND").unwrap_or_else(|_| "auto".to_string());
    BackendPreference::parse(&raw).ok_or_else(|| {
        Error::InvalidInput(format!(
            "Unsupported IZWI_FISH_S2_BACKEND `{raw}`; expected auto, cpu, metal, or cuda"
        ))
    })
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<usize>()
            .map_err(|err| Error::InvalidInput(format!("Invalid {name} value `{raw}`: {err}"))),
        Err(_) => Ok(default),
    }
}

fn env_f32(name: &str, default: f32) -> Result<f32> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<f32>()
            .map_err(|err| Error::InvalidInput(format!("Invalid {name} value `{raw}`: {err}"))),
        Err(_) => Ok(default),
    }
}

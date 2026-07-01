use std::env;
use std::path::{Path, PathBuf};

use crate::backends::{BackendPreference, DeviceSelector};
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

use super::{FishS2DacConfig, FishS2GenerationParams, FishS2Reference, FishS2TtsModel};

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

#[test]
#[ignore = "requires local fishaudio/s2-pro artifacts and a short reference WAV"]
fn fish_s2_real_model_smoke_generates_finite_audio() -> Result<()> {
    let model_dir = required_env_path("IZWI_FISH_S2_MODEL_DIR")?;
    let reference_wav = required_env_path("IZWI_FISH_S2_REFERENCE_WAV")?;
    let reference_text = env::var("IZWI_FISH_S2_REFERENCE_TEXT")
        .unwrap_or_else(|_| "This is the reference voice for a short smoke test.".to_string());
    let target_text = env::var("IZWI_FISH_S2_TARGET_TEXT")
        .unwrap_or_else(|_| "This is a short Fish Audio S2 smoke test.".to_string());
    let max_frames = env_usize("IZWI_FISH_S2_SMOKE_MAX_FRAMES", 24)?;
    let temperature = env_f32("IZWI_FISH_S2_SMOKE_TEMPERATURE", 0.0)?;
    let top_p = env_f32("IZWI_FISH_S2_SMOKE_TOP_P", 1.0)?;
    let backend = env_backend()?;
    let device = DeviceSelector::detect_for_preference(backend)?;
    let (audio_samples, sample_rate) = read_wav_mono(&reference_wav)?;

    let model = FishS2TtsModel::load(&model_dir, ModelVariant::FishAudioS2Pro, device)?;
    let output = model.generate_with_reference(
        &target_text,
        FishS2Reference {
            audio_samples,
            sample_rate,
            text: reference_text,
        },
        FishS2GenerationParams {
            max_frames,
            temperature,
            top_p,
        },
    )?;

    let dac_config = FishS2DacConfig::current();
    assert_eq!(output.sample_rate, dac_config.sample_rate);
    assert!(output.frames_generated > 0);
    assert!(output.samples.len() >= dac_config.samples_per_frame()?);
    assert!(output.samples.iter().all(|sample| sample.is_finite()));
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

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path).map_err(|err| {
        Error::AudioError(format!(
            "Failed to open Fish S2 reference WAV {}: {err}",
            path.display()
        ))
    })?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let sample_rate = spec.sample_rate;
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| {
                sample.map_err(|err| {
                    Error::AudioError(format!(
                        "Failed to read float WAV sample from {}: {err}",
                        path.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => {
            let scale = (1_i64 << spec.bits_per_sample.saturating_sub(1)) as f32;
            reader
                .samples::<i16>()
                .map(|sample| {
                    sample
                        .map(|value| (f32::from(value) / scale).clamp(-1.0, 1.0))
                        .map_err(|err| {
                            Error::AudioError(format!(
                                "Failed to read int16 WAV sample from {}: {err}",
                                path.display()
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?
        }
        hound::SampleFormat::Int => {
            let scale = (1_i64 << spec.bits_per_sample.saturating_sub(1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| ((value as f32) / scale).clamp(-1.0, 1.0))
                        .map_err(|err| {
                            Error::AudioError(format!(
                                "Failed to read int32 WAV sample from {}: {err}",
                                path.display()
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    if interleaved.is_empty() {
        return Err(Error::AudioError(format!(
            "Fish S2 reference WAV is empty: {}",
            path.display()
        )));
    }

    let mut mono = Vec::with_capacity(interleaved.len() / channels.max(1));
    for frame in interleaved.chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / frame.len() as f32);
    }
    Ok((mono, sample_rate))
}

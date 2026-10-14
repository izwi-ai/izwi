//! Explicitly invoked, real-weight intelligibility check of VibeVoice TTS
//! through the runtime the server uses: a voice-cloned synthesis, written as
//! a WAV, transcribed by Parakeet-TDT-0.6B-v3 on the CPU.
use std::env;
use std::path::PathBuf;

use base64::Engine as _;

use crate::backends::{BackendPreference, DeviceProfile};
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::models::architectures::parakeet::asr::ParakeetAsrModel;
use crate::runtime::{GenerationRequest, RuntimeService};
use crate::{ContextLengthPreference, EngineConfig};

const TARGET: &str = "The quick brown fox jumps over the lazy dog.";

fn required(name: &str) -> Result<String> {
    env::var(name).map_err(|_| Error::InvalidInput(format!("{name} is required")))
}

/// Lower-case words, punctuation dropped.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word error rate of `hypothesis` against `reference` (Levenshtein over
/// words).
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let (reference, hypothesis) = (words(reference), words(hypothesis));
    let mut row = (0..=hypothesis.len()).collect::<Vec<_>>();
    for (i, r) in reference.iter().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let substitution = previous + usize::from(r != h);
            previous = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[hypothesis.len()] as f64 / reference.len().max(1) as f64
}

/// Synthesizes [`TARGET`] (or `IZWI_VIBEVOICE_TEXT`) in the voice of
/// `IZWI_VIBEVOICE_REFERENCE_WAV` on `IZWI_VIBEVOICE_BACKEND` (cpu or
/// metal), writes `<IZWI_VIBEVOICE_OUT_DIR>/<backend>.wav` and requires
/// Parakeet's transcript to be within `IZWI_VIBEVOICE_MAX_WER` (0.34) of
/// the text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires VibeVoice-1.5B and Parakeet-TDT-0.6B-v3 weights and IZWI_VIBEVOICE_* settings"]
async fn vibevoice_real_clone_is_intelligible() -> Result<()> {
    let variant = ModelVariant::VibeVoice15BTts;
    let models_dir = PathBuf::from(required("IZWI_MODELS_DIR")?);
    let backend_name = required("IZWI_VIBEVOICE_BACKEND")?;
    let backend = match backend_name.as_str() {
        "cpu" => BackendPreference::Cpu,
        "metal" => BackendPreference::Metal,
        "cuda" => BackendPreference::Cuda,
        _ => {
            return Err(Error::InvalidInput(
                "backend must be cpu, metal or cuda".into(),
            ))
        }
    };
    let reference = PathBuf::from(required("IZWI_VIBEVOICE_REFERENCE_WAV")?);
    let reference_text = required("IZWI_VIBEVOICE_REFERENCE_TEXT")?;
    let out_dir = PathBuf::from(required("IZWI_VIBEVOICE_OUT_DIR")?);
    let text = env::var("IZWI_VIBEVOICE_TEXT").unwrap_or_else(|_| TARGET.to_string());
    let max_wer = env::var("IZWI_VIBEVOICE_MAX_WER")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(0.34);

    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.clone(),
        backend,
        max_sequence_length: ContextLengthPreference::explicit(4096)?,
        ..EngineConfig::default()
    })?;
    runtime.load_model(variant).await?;
    let mut request = GenerationRequest::new(text.as_str()).with_model_variant(variant);
    request.reference_audio =
        Some(base64::engine::general_purpose::STANDARD.encode(std::fs::read(&reference)?));
    request.reference_text = Some(reference_text);
    let result = runtime.generate(request).await?;
    drop(runtime);

    std::fs::create_dir_all(&out_dir)?;
    let wav = out_dir.join(format!("{backend_name}.wav"));
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: result.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .map_err(|error| Error::InferenceError(error.to_string()))?;
    for sample in &result.samples {
        writer
            .write_sample(*sample)
            .map_err(|error| Error::InferenceError(error.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|error| Error::InferenceError(error.to_string()))?;

    let asr = ParakeetAsrModel::load(
        &models_dir.join(ModelVariant::ParakeetTdt06BV3.dir_name()),
        ModelVariant::ParakeetTdt06BV3,
        DeviceProfile::cpu(),
    )?;
    let transcript = asr.transcribe(&result.samples, result.sample_rate, Some("en"))?;
    let wer = word_error_rate(&text, &transcript);
    let rms = (result.samples.iter().map(|s| s * s).sum::<f32>()
        / result.samples.len().max(1) as f32)
        .sqrt();
    eprintln!(
        "VibeVoice intelligibility: {}",
        serde_json::json!({
            "backend": backend_name,
            "wav": wav,
            "frames": result.total_tokens,
            "stop_reason": result.stop_reason.map(|reason| format!("{reason:?}")),
            "seconds": result.duration_secs(),
            "rms": rms,
            "text": text,
            "transcript": transcript,
            "wer": wer,
        })
    );
    assert!(wer <= max_wer, "WER {wer:.3}: {transcript:?}");
    Ok(())
}

#[test]
fn word_error_rate_counts_word_edits() {
    assert_eq!(word_error_rate("The quick fox.", "the quick fox"), 0.0);
    assert!((word_error_rate("a b c", "a x c d") - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(word_error_rate("a b", ""), 1.0);
}

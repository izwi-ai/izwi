//! SOV-Transcribe-1 through the runtime on the synthetic checkpoint
//! (`sov_transcribe::synthetic`, with published 30 s windows): the
//! CUDA-only gate, load admission, retained rows with resumable prefill
//! spans, continuous decode across concurrent requests, long files as chunk
//! rows, and request validation. The CPU runs the synthetic checkpoint as
//! the test reference; the published model loads on CUDA only.

use super::*;
use crate::config::EngineConfig;
use crate::model::ModelVariant;
use crate::models::architectures::sov_transcribe::synthetic::{
    write_checkpoint_with, PUBLISHED_POSITIONS,
};

const VARIANT: ModelVariant = ModelVariant::SovTranscribe1;

/// `data/fox.wav`'s sample rate.
const FOX_RATE: u32 = 24_000;
/// The synthetic model decodes random words until its budget; keep it short.
const MAX_TOKENS: usize = 16;

fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: FOX_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for sample in samples {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * 32_767.0) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

/// Real speech (`data/fox.wav`), so the no-speech gate lets the request
/// reach the model: `seconds` of it from `start`, looped as needed.
fn fox(start: f32, seconds: f32) -> Vec<f32> {
    let bytes =
        std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav"))
            .expect("data/fox.wav");
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(reader.spec().sample_rate, FOX_RATE);
    let samples = reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
        .collect::<Vec<_>>();
    let rate = FOX_RATE as f32;
    let first = (start * rate) as usize;
    (0..(seconds * rate) as usize)
        .map(|i| samples[(first + i) % samples.len()])
        .collect()
}

struct ModelsDir(std::path::PathBuf);

impl Drop for ModelsDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn models_dir(tag: &str, with_checkpoint: bool) -> ModelsDir {
    let root = std::env::temp_dir().join(format!(
        "izwi-sov-transcribe-runtime-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    if with_checkpoint {
        write_checkpoint_with(&root.join(VARIANT.dir_name()), PUBLISHED_POSITIONS);
    }
    ModelsDir(root)
}

fn runtime_without_model(models: &ModelsDir, rows: usize) -> RuntimeService {
    RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend: crate::backends::BackendPreference::Cpu,
        max_batch_size: crate::config::BatchSizePreference::Fixed(
            std::num::NonZeroUsize::new(rows).unwrap(),
        ),
        ..Default::default()
    })
    .expect("runtime")
}

async fn runtime_with(models: &ModelsDir, rows: usize) -> RuntimeService {
    let runtime = runtime_without_model(models, rows);
    runtime
        .load_model(VARIANT)
        .await
        .expect("load the synthetic checkpoint");
    runtime
}

async fn transcribe_with(
    runtime: &RuntimeService,
    clip: &[u8],
    language: Option<&str>,
    prompt: Option<&str>,
    keywords: &[&str],
) -> crate::error::Result<AsrTranscription> {
    runtime
        .asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            clip,
            Some(&VARIANT.to_string()),
            language,
            prompt,
            crate::engine::AsrTaskOptions::new(
                None,
                keywords.iter().map(|keyword| keyword.to_string()).collect(),
            ),
            Some(MAX_TOKENS),
            None,
            RuntimeRequestContext::default(),
            false,
        )
        .await
}

async fn transcribe(
    runtime: &RuntimeService,
    clip: &[u8],
) -> crate::error::Result<AsrTranscription> {
    transcribe_with(runtime, clip, None, None, &[]).await
}

async fn family_diagnostics(runtime: &RuntimeService) -> serde_json::Value {
    runtime
        .loaded_model_diagnostics()
        .await
        .into_iter()
        .find_map(|loaded| loaded.family_diagnostics)
        .expect("the model's diagnostics")
}

/// The published model on a CPU server: listed by the catalogue, and every
/// way to load it refused with the one message, before a download or a read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_model_is_refused_on_a_backend_it_cannot_run_on() {
    let models = models_dir("gate", false);
    let runtime = runtime_without_model(&models, 1);
    assert!(ModelVariant::all().contains(&VARIANT));
    let expected = "SOV-Transcribe-1 needs an NVIDIA GPU (CUDA); this server runs on the CPU";
    let error = runtime
        .ensure_model_runs_on_this_backend(VARIANT)
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::InvalidInput);
    assert!(error.to_string().contains(expected), "{error}");
    let error = runtime.load_model(VARIANT).await.unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::InvalidInput);
    assert!(error.to_string().contains(expected), "{error}");
    // A transcription request for it fails the same way; nothing loads.
    let clip = wav_bytes(&fox(0.0, 3.0));
    let error = transcribe(&runtime, &clip).await.unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::InvalidInput);
    assert!(error.to_string().contains(expected), "{error}");
    assert!(runtime.loaded_model_diagnostics().await.is_empty());
    // Models without a required backend are not affected.
    runtime
        .ensure_model_runs_on_this_backend(ModelVariant::WhisperLargeV3Turbo)
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_rows_transcribe_alone_and_concurrently_alike() {
    let models = models_dir("rows", true);
    let runtime = runtime_with(&models, 3).await;
    let clips = [
        wav_bytes(&fox(0.0, 5.0)),
        wav_bytes(&fox(1.0, 9.0)),
        wav_bytes(&fox(0.5, 3.0)),
    ];
    let mut solo = Vec::new();
    for clip in &clips {
        let transcription = transcribe(&runtime, clip).await.expect("transcribe");
        let diagnostics = transcription.asr_diagnostics.expect("diagnostics");
        assert_eq!(diagnostics["family"], "sov_transcribe_asr");
        // One 30 s window whatever the clip's length: 375 audio tokens.
        assert_eq!(diagnostics["encoder_windows"], 1);
        assert_eq!(diagnostics["audio_tokens"], 375);
        solo.push(transcription.text);
    }
    eprintln!("synthetic transcripts: {solo:?}");
    assert!(solo.iter().all(|text| !text.is_empty()), "{solo:?}");
    let again = transcribe(&runtime, &clips[0]).await.expect("again");
    assert_eq!(again.text, solo[0]);
    let concurrent =
        futures::future::join_all(clips.iter().map(|clip| transcribe(&runtime, clip))).await;
    for (result, expected) in concurrent.into_iter().zip(&solo) {
        assert_eq!(&result.expect("concurrent transcribe").text, expected);
    }
    let diagnostics = family_diagnostics(&runtime).await;
    eprintln!("diagnostics: {diagnostics}");
    assert_eq!(diagnostics["device_kind"], "cpu");
    assert_eq!(diagnostics["batched_prefill"], true);
    assert!(diagnostics["windows_encoded"].as_u64().unwrap() >= 7);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_requests_are_validated_before_admission() {
    let models = models_dir("validation", true);
    let runtime = runtime_with(&models, 1).await;
    let clip = wav_bytes(&fox(0.0, 3.0));
    // A language outside the model's seven fails as invalid input.
    for language in ["ja", "tlh"] {
        let error = transcribe_with(&runtime, &clip, Some(language), None, &[])
            .await
            .unwrap_err();
        assert!(
            matches!(error, crate::error::Error::InvalidInput(_)),
            "{language}: {error}"
        );
    }
    // A supported language, `auto`, keywords and an instruction all run,
    // and the instruction the model saw is reported.
    for (language, prompt, keywords, expected) in [
        (
            Some("de"),
            None,
            &[][..],
            "Transcribe the German audio into text, ensuring all punctuation marks are included.",
        ),
        (
            Some("auto"),
            None,
            &["Izwi", "Lovelace"][..],
            "This audio may contain the following words or phrases: Izwi, Lovelace. Transcribe the audio into text.",
        ),
        (None, Some("Write numbers as digits."), &[][..], "Write numbers as digits."),
    ] {
        let transcription = transcribe_with(&runtime, &clip, language, prompt, keywords)
            .await
            .expect("transcribe");
        assert_eq!(
            transcription.asr_diagnostics.expect("diagnostics")["instruction"],
            expected
        );
    }
    // The model neither translates nor tags speakers.
    let error = runtime
        .asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            &clip,
            Some(&VARIANT.to_string()),
            None,
            None,
            crate::engine::AsrTaskOptions::new(Some("en".into()), Vec::new()),
            Some(MAX_TOKENS),
            None,
            RuntimeRequestContext::default(),
            false,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::error::Error::InvalidInput(_)),
        "{error}"
    );
}

/// A file longer than one window is cut on speech and its chunks run as
/// rows: every chunk is one 30 s window, and the texts join in order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_long_file_runs_as_chunk_rows() {
    let models = models_dir("long", true);
    let runtime = runtime_with(&models, 3).await;
    let long = wav_bytes(&fox(0.0, 75.0));
    let before = family_diagnostics(&runtime).await["windows_encoded"]
        .as_u64()
        .unwrap();
    let transcription = transcribe(&runtime, &long).await.expect("long transcribe");
    assert!(!transcription.text.is_empty());
    assert!((transcription.duration_secs - 75.0).abs() < 1.0);
    let diagnostics = family_diagnostics(&runtime).await;
    let windows = diagnostics["windows_encoded"].as_u64().unwrap() - before;
    // 75 s in chunks of at most 30 s: three rows or more, one window each.
    assert!(windows >= 3, "{windows} windows");
    eprintln!("long file: {windows} chunk rows, {:?}", transcription.text);
}

/// Several decode steps per engine quantum (the default while nothing
/// waits) transcribe exactly as one step per quantum, and short prefill
/// spans as one span.
// The env lock is held across awaits on purpose: it keeps the variables
// fixed while each runtime loads.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_quanta_and_spans_do_not_change_a_transcript() {
    use crate::models::architectures::sov_transcribe::{DECODE_QUANTUM_ENV, PREFILL_SPAN_ENV};
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let models = models_dir("quanta", true);
    let clip = wav_bytes(&fox(0.0, 5.0));
    let mut texts = Vec::new();
    // Read at load: one runtime per setting.
    for (steps, span) in [("1", "64"), ("4", "192"), ("7", "16")] {
        std::env::set_var(DECODE_QUANTUM_ENV, steps);
        std::env::set_var(PREFILL_SPAN_ENV, span);
        let runtime = runtime_with(&models, 1).await;
        texts.push(transcribe(&runtime, &clip).await.expect("transcribe").text);
    }
    std::env::remove_var(DECODE_QUANTUM_ENV);
    std::env::remove_var(PREFILL_SPAN_ENV);
    assert!(!texts[0].is_empty());
    assert_eq!(texts[1], texts[0]);
    assert_eq!(texts[2], texts[0]);
}

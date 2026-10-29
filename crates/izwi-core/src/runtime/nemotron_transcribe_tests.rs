//! Nemotron 3.5 Transcribe through the runtime on the synthetic checkpoint
//! (`nemotron_transcribe::synthetic`): load admission, retained rows with
//! resumable prefill spans, continuous decode across concurrent requests,
//! and request validation, on the CPU. The published checkpoint runs the
//! same routes in `nemotron_transcribe::reference_tests` and the e2e kit.

use super::*;
use crate::config::EngineConfig;
use crate::model::ModelVariant;
use crate::models::architectures::nemotron_transcribe::synthetic::write_checkpoint;

const VARIANT: ModelVariant = ModelVariant::Nemotron35Transcribe30BA4BFp8;

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

/// The synthetic model decodes random words until its budget; keep it short.
const MAX_TOKENS: usize = 24;

async fn transcribe(
    runtime: &RuntimeService,
    clip: &[u8],
    language: Option<&str>,
) -> crate::error::Result<AsrTranscription> {
    runtime
        .asr_transcribe_bytes_with_prompt_max_tokens_and_correlation(
            clip,
            Some(&VARIANT.to_string()),
            language,
            None,
            Some(MAX_TOKENS),
            None,
        )
        .await
}

/// `data/fox.wav`'s sample rate.
const FOX_RATE: u32 = 24_000;

/// Real speech (`data/fox.wav`), so the no-speech gate lets the request
/// reach the model: `seconds` of it from `start`, looped as needed. The
/// runtime resamples it to 16 kHz.
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

fn synthetic_models_dir(tag: &str) -> ModelsDir {
    let root = std::env::temp_dir().join(format!(
        "izwi-nemotron-transcribe-runtime-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    write_checkpoint(&root.join(VARIANT.dir_name()));
    ModelsDir(root)
}

async fn runtime_with(models: &ModelsDir, rows: usize) -> RuntimeService {
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend: crate::backends::BackendPreference::Cpu,
        max_batch_size: crate::config::BatchSizePreference::Fixed(
            std::num::NonZeroUsize::new(rows).unwrap(),
        ),
        ..Default::default()
    })
    .expect("runtime");
    runtime
        .load_model(VARIANT)
        .await
        .expect("load the synthetic checkpoint");
    runtime
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_rows_transcribe_alone_and_concurrently_alike() {
    // Prefill in 16-token spans, so every prompt resumes across several
    // quanta (the published model's spans are 512 or 2,048 tokens).
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous =
        std::env::var_os(crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV);
    std::env::set_var(
        crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV,
        "16",
    );
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var(
                    crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV,
                    value,
                ),
                None => std::env::remove_var(
                    crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV,
                ),
            }
        }
    }
    let _restore = Restore(previous);
    let models = synthetic_models_dir("rows");
    let runtime = runtime_with(&models, 3).await;
    let clips = [
        wav_bytes(&fox(0.0, 5.0)),
        wav_bytes(&fox(1.0, 9.0)),
        wav_bytes(&fox(0.5, 3.0)),
    ];
    let mut solo = Vec::new();
    for clip in &clips {
        let transcription = transcribe(&runtime, clip, None).await.expect("transcribe");
        solo.push(transcription.text);
    }
    eprintln!("synthetic transcripts: {solo:?}");
    // Random weights decode words, not the stop token at once: the
    // comparisons below are over real text.
    assert!(solo.iter().all(|text| !text.is_empty()), "{solo:?}");
    // Deterministic, and the same beside other rows (bit-identical decode
    // steps make the same tokens).
    let again = transcribe(&runtime, &clips[0], None)
        .await
        .expect("transcribe again");
    assert_eq!(again.text, solo[0]);
    let concurrent =
        futures::future::join_all(clips.iter().map(|clip| transcribe(&runtime, clip, None))).await;
    for (result, expected) in concurrent.into_iter().zip(&solo) {
        assert_eq!(&result.expect("concurrent transcribe").text, expected);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_requests_are_validated_before_admission() {
    let models = synthetic_models_dir("validation");
    let runtime = runtime_with(&models, 1).await;
    let clip = wav_bytes(&fox(0.0, 3.0));
    // A language the card lists without ASR, and an unknown one, fail as
    // invalid input (not as a missing model).
    for language in ["no", "tlh"] {
        let error = transcribe(&runtime, &clip, Some(language))
            .await
            .unwrap_err();
        assert!(
            matches!(error, crate::error::Error::InvalidInput(_)),
            "{language}: {error}"
        );
    }
    // A supported language and `auto` run.
    transcribe(&runtime, &clip, Some("de"))
        .await
        .expect("German");
    transcribe(&runtime, &clip, Some("auto"))
        .await
        .expect("auto");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_single_row_resumes_prefill_across_spans() {
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::env::set_var(
        crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV,
        "16",
    );
    let models = synthetic_models_dir("spans");
    let runtime = runtime_with(&models, 1).await;
    std::env::remove_var(crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV);
    let clip = wav_bytes(&fox(0.0, 5.0));
    let started = std::time::Instant::now();
    let text = transcribe(&runtime, &clip, None)
        .await
        .expect("transcribe")
        .text;
    eprintln!("spans: {text:?} in {:?}", started.elapsed());
    assert!(!text.is_empty());
}

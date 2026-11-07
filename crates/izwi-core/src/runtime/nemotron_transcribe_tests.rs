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
    runtime_on(models, rows, crate::backends::BackendPreference::Cpu).await
}

async fn runtime_on(
    models: &ModelsDir,
    rows: usize,
    backend: crate::backends::BackendPreference,
) -> RuntimeService {
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend,
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

// The env lock is held across awaits on purpose: it keeps the span
// variable fixed for the whole test.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_rows_transcribe_alone_and_concurrently_alike() {
    // Prefill in 16-token spans, so every prompt resumes across several
    // quanta (the published model's spans are 192 or 512 tokens).
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
    // Packed prefill is on; whether the concurrent rows shared a prefill
    // quantum depends on when each arrived (most runs pack several).
    let packed = packed_prefill(&runtime).await;
    eprintln!("packed prefill: {packed}");
    assert_eq!(packed["on"], true);
    assert!(packed["packed_rows"].as_u64() >= packed["packed_quanta"].as_u64().map(|q| 2 * q));
}

/// The model's packed-prefill diagnostics (`batched_prefill`).
async fn packed_prefill(runtime: &RuntimeService) -> serde_json::Value {
    runtime
        .loaded_model_diagnostics()
        .await
        .into_iter()
        .find_map(|loaded| loaded.family_diagnostics)
        .map(|diagnostics| diagnostics["batched_prefill"].clone())
        .expect("the model's diagnostics")
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

// The env lock is held across awaits on purpose: it keeps the span
// variable fixed for the whole test.
#[allow(clippy::await_holding_lock)]
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

/// Several decode steps per engine quantum (the default while nothing
/// waits) transcribe exactly as one step per quantum.
// The env lock is held across awaits on purpose: it keeps the quantum
// variable fixed while each runtime loads.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_decode_quanta_transcribe_as_single_steps() {
    use crate::models::architectures::nemotron_transcribe::DECODE_QUANTUM_ENV;
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let models = synthetic_models_dir("quanta");
    let clip = wav_bytes(&fox(0.0, 5.0));
    let mut texts = Vec::new();
    // Read at load: one runtime per quantum.
    for steps in ["1", "4", "7"] {
        std::env::set_var(DECODE_QUANTUM_ENV, steps);
        let runtime = runtime_with(&models, 1).await;
        texts.push(
            transcribe(&runtime, &clip, None)
                .await
                .expect("transcribe")
                .text,
        );
    }
    std::env::remove_var(DECODE_QUANTUM_ENV);
    assert!(!texts[0].is_empty());
    assert_eq!(texts[1], texts[0]);
    assert_eq!(texts[2], texts[0]);
}

/// The same rows on Metal (FP8 weights, F32 activations, F16 KV): load,
/// resumable spans, and two concurrent rows transcribing as they do alone.
#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_rows_transcribe_alike_on_metal() {
    let Some(_metal) =
        crate::backends::metal_test_device_or_skip("synthetic_rows_transcribe_alike_on_metal")
    else {
        return;
    };
    let models = synthetic_models_dir("metal");
    let runtime = runtime_on(&models, 2, crate::backends::BackendPreference::Metal).await;
    let clips = [wav_bytes(&fox(0.0, 5.0)), wav_bytes(&fox(1.0, 7.0))];
    let mut solo = Vec::new();
    for clip in &clips {
        solo.push(
            transcribe(&runtime, clip, None)
                .await
                .expect("transcribe")
                .text,
        );
    }
    eprintln!("synthetic Metal transcripts: {solo:?}");
    assert!(solo.iter().all(|text| !text.is_empty()), "{solo:?}");
    let concurrent =
        futures::future::join_all(clips.iter().map(|clip| transcribe(&runtime, clip, None))).await;
    for (result, expected) in concurrent.into_iter().zip(&solo) {
        assert_eq!(&result.expect("concurrent transcribe").text, expected);
    }
    eprintln!("packed prefill: {}", packed_prefill(&runtime).await);
}

/// Word timestamps through the transcription route: the CTC stage runs on
/// the row and its words reach `timestamp_words` (tags left out) and the
/// full `ctc_timestamps` object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_word_timestamps_reach_the_diagnostics() {
    let models = synthetic_models_dir("timestamps");
    let runtime = runtime_with(&models, 1).await;
    let clip = wav_bytes(&fox(0.0, 4.0));
    let output = runtime
        .asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            &clip,
            Some(&VARIANT.to_string()),
            None,
            None,
            crate::engine::AsrTaskOptions::default(),
            Some(MAX_TOKENS),
            None,
            RuntimeRequestContext::default(),
            true,
        )
        .await
        .expect("transcribe with timestamps");
    let diagnostics = output.asr_diagnostics.expect("diagnostics");
    assert!(
        diagnostics["ctc_timestamps"]["error"].is_null(),
        "{diagnostics}"
    );
    // The row's stage timing reaches the diagnostics, the CTC stage included.
    let timing = &diagnostics["timing"];
    assert!(timing["prefill_spans"].as_u64().unwrap() >= 1, "{timing}");
    assert!(timing["ctc_ms"].as_f64().unwrap() > 0.0, "{timing}");
    let words = diagnostics["timestamp_words"]
        .as_array()
        .expect("timestamp_words");
    assert!(!words.is_empty(), "{diagnostics}");
    for word in words {
        let text = word["word"].as_str().unwrap();
        assert!(!text.starts_with('<'), "{word}");
        let (start, end) = (
            word["start"].as_f64().unwrap(),
            word["end"].as_f64().unwrap(),
        );
        assert!(0.0 <= start && start <= end && end <= 4.0 + 1e-6, "{word}");
    }
    // Without the flag, no CTC stage.
    let plain = transcribe(&runtime, &clip, None).await.expect("transcribe");
    assert!(plain
        .asr_diagnostics
        .is_none_or(|d| d.get("ctc_timestamps").is_none()));
}

/// Speaker-attributed ASR and one-model diarization run on the model's own
/// rows: one transcription with speaker tags and word timestamps each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_speaker_attribution_and_one_model_diarization() {
    let models = synthetic_models_dir("speakers");
    let runtime = runtime_with(&models, 1).await;
    let clip = wav_bytes(&fox(0.0, 4.0));
    let started = std::time::Instant::now();
    let saa = runtime
        .speaker_attributed_asr_bytes(&clip, Some(&VARIANT.to_string()), Some("en"), None, None)
        .await
        .expect("speaker-attributed ASR");
    eprintln!("SAA in {:?}: {saa:?}", started.elapsed());
    assert!(!saa.text.is_empty());
    assert!(!saa.speaker_turns.is_empty());
    let config = crate::runtime::DiarizationConfig::default();
    let options = crate::runtime::DiarizationPipelineOptions::default();
    // A pipeline ASR beside a model that diarizes on its own is refused.
    let error = runtime
        .diarize_with_transcript_bytes(
            &clip,
            Some(&VARIANT.to_string()),
            Some("parakeet-tdt-0.6b-v3"),
            None,
            &config,
            &options,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::error::Error::InvalidInput(_)),
        "{error}"
    );
    let started = std::time::Instant::now();
    let diarized = runtime
        .diarize_with_transcript_bytes(
            &clip,
            Some(&VARIANT.to_string()),
            None,
            None,
            &config,
            &options,
        )
        .await
        .expect("one-model diarization");
    eprintln!("diarization in {:?}: {diarized:?}", started.elapsed());
    assert!(!diarized.asr_text.is_empty());
    // The random model may write more tokens than its CTC timeline holds;
    // then the job keeps the transcript and says why it has no words.
    if diarized.words.is_empty() {
        assert!(!diarized.warnings.is_empty(), "{diarized:?}");
    }
    assert!(diarized
        .words
        .iter()
        .all(|word| word.speaker.starts_with("SPEAKER_") || word.speaker == "UNKNOWN"));
    // Decoder special tokens never reach the text.
    assert!(
        !saa.text.contains("<|im_start|>") && !saa.text.contains("<reserved_"),
        "{}",
        saa.text
    );
}

/// A translation longer than one segment runs as segments: its text is the
/// in-order join of each segment translated alone, its stream's deltas and
/// progress follow the segments in order, and a transcription of the same
/// audio stays one pass.
// The env lock is held across awaits on purpose: it keeps the span
// variable fixed for the whole test.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_long_translations_run_in_segments() {
    use super::asr::{append_segment, plan_translation_segments, segment_wav};
    use crate::engine::{AsrProgress, AsrProgressPhase, AsrTaskOptions};
    use crate::models::architectures::nemotron_transcribe::PREFILL_SPAN_ENV;
    use std::sync::{Arc, Mutex};

    // The default span grid (192 tokens on the CPU): a segment prefills in
    // the same spans alone and beside another row, so it translates alike.
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var(PREFILL_SPAN_ENV, value),
                None => std::env::remove_var(PREFILL_SPAN_ENV),
            }
        }
    }
    let _restore = Restore(std::env::var_os(PREFILL_SPAN_ENV));
    std::env::remove_var(PREFILL_SPAN_ENV);
    let models = synthetic_models_dir("translation-segments");
    // Two rows, so the segments both share a batch and queue for one.
    let runtime = runtime_with(&models, 2).await;
    let model = VARIANT.to_string();
    let clip = wav_bytes(&fox(0.0, 70.0));
    let translate = AsrTaskOptions::new(Some("de".into()), Vec::new());
    let run = |audio: Vec<u8>, task: AsrTaskOptions| {
        let (runtime, model) = (&runtime, &model);
        async move {
            runtime
                .asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
                    &audio,
                    Some(model),
                    None,
                    None,
                    task,
                    Some(MAX_TOKENS),
                    None,
                    RuntimeRequestContext::default(),
                    false,
                )
                .await
                .expect("ASR request")
        }
    };

    let translation = run(clip.clone(), translate.clone()).await;
    assert_eq!(translation.language.as_deref(), Some("de"));
    let chunking = &translation.asr_diagnostics.as_ref().expect("diagnostics")["chunking"];
    assert_eq!(chunking["planner"], "translation_segments", "{chunking}");
    let (samples, rate) =
        crate::runtime::audio_io::decode_audio_bytes_canonical(&clip).expect("decode");
    let segments = plan_translation_segments(&samples, rate, 30.0);
    assert!(segments.len() >= 3, "{segments:?}");
    assert_eq!(chunking["chunk_count"], segments.len());

    let mut expected = String::new();
    for segment in &segments {
        let audio = segment_wav(&samples[segment.start_sample..segment.end_sample], rate).unwrap();
        let alone = run(audio, translate.clone()).await;
        assert!(alone.asr_diagnostics.as_ref().unwrap()["chunking"].is_null());
        append_segment(&mut expected, &alone.text);
    }
    eprintln!("segmented translation: {:?}", translation.text);
    assert!(!expected.is_empty());
    assert_eq!(translation.text, expected);

    // The stream emits each segment's text in order, ending with the whole.
    let deltas = Arc::new(Mutex::new(Vec::new()));
    let sink = deltas.clone();
    let streamed = runtime
        .asr_transcribe_streaming_bytes_with_runtime_context(
            &clip,
            Some(&model),
            None,
            None,
            translate.clone(),
            Some(MAX_TOKENS),
            None,
            RuntimeRequestContext::default(),
            move |delta| sink.lock().unwrap().push(delta),
        )
        .await
        .expect("streamed translation");
    assert_eq!(streamed.text, expected);
    assert_eq!(deltas.lock().unwrap().concat(), expected);

    // Progress counts finished segments up to the whole recording.
    let progress = Arc::new(Mutex::new(Vec::<AsrProgress>::new()));
    let events = progress.clone();
    runtime
        .asr_transcribe_bytes_with_progress_and_runtime_context(
            &clip,
            Some(&model),
            None,
            translate,
            None,
            RuntimeRequestContext::default(),
            false,
            |_| {},
            move |event| events.lock().unwrap().push(event),
        )
        .await
        .expect("translation with progress");
    let progress = progress.lock().unwrap();
    let finished = progress
        .iter()
        .filter(|event| event.phase == AsrProgressPhase::ChunkFinished)
        .map(|event| event.current_chunk)
        .collect::<Vec<_>>();
    assert_eq!(
        finished,
        (1..=segments.len()).map(Some).collect::<Vec<_>>(),
        "{progress:?}"
    );
    assert_eq!(progress.last().unwrap().phase, AsrProgressPhase::Complete);

    // A transcription of the same recording stays one pass.
    let transcript = run(clip, AsrTaskOptions::default()).await;
    assert!(transcript.asr_diagnostics.as_ref().unwrap()["chunking"].is_null());
}

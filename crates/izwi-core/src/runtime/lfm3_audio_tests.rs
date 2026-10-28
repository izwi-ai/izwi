//! LFM3-Audio through the runtime: load, then every public speech route
//! the server calls, on the backend the runtime selects.
//!
//! The test needs the published GGUFs. It runs when
//! `IZWI_LFM3_AUDIO_MODELS_DIR` names the models directory and is otherwise
//! reported as not run.
//!
//! `IZWI_LFM3_AUDIO_BACKEND=metal` runs it on Metal instead of the CPU, and
//! `IZWI_LFM3_AUDIO_BATCH=4` runs speech and transcription rows four wide.
//!
//! Run: `IZWI_LFM3_AUDIO_MODELS_DIR=~/Library/Application\ Support/izwi/models cargo test -p izwi-core --features metal --release --lib lfm3_audio_tests -- --nocapture`

use std::io::Write;
use std::time::Instant;

use tokio::sync::mpsc;

use super::*;
use crate::config::EngineConfig;
use crate::model::ModelVariant;
use crate::GenerationParams;

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog.";

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// This process's resident memory, from `ps`.
fn resident_bytes() -> u64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok())
        .map_or(0, |kib| kib * 1024)
}

/// A 16-bit mono WAV's samples and rate.
fn wav_samples(bytes: &[u8]) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
    let sample_rate = reader.spec().sample_rate;
    let samples = reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
        .collect();
    (samples, sample_rate)
}

fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lfm3_audio_serves_asr_tts_and_voice_chat_if_env_set() {
    let Some(models_dir) = std::env::var_os("IZWI_LFM3_AUDIO_MODELS_DIR") else {
        let _ = writeln!(
            std::io::stderr(),
            "IZWI_LFM3_AUDIO_MODELS_DIR unset; not run"
        );
        return;
    };
    // Tests default to the CPU backend; `IZWI_LFM3_AUDIO_BACKEND=metal`
    // (or `cuda`) runs the same routes on a device.
    let backend = std::env::var("IZWI_LFM3_AUDIO_BACKEND")
        .ok()
        .and_then(|name| crate::backends::BackendPreference::parse(&name))
        .unwrap_or(crate::backends::BackendPreference::Cpu);
    #[cfg(feature = "metal")]
    let _metal_lease = (backend == crate::backends::BackendPreference::Metal).then(|| {
        crate::backends::metal_test_device_or_skip(
            "lfm3_audio_serves_asr_tts_and_voice_chat_if_env_set",
        )
    });
    // Speech rows run two wide on CPU and Metal (the portable certified
    // width); `IZWI_LFM3_AUDIO_BATCH=4` sets an explicit width, as an
    // operator's `IZWI_MAX_BATCH_SIZE` does.
    let max_batch_size = std::env::var("IZWI_LFM3_AUDIO_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(std::num::NonZeroUsize::new)
        .map_or(crate::config::BatchSizePreference::Auto, |rows| {
            crate::config::BatchSizePreference::Fixed(rows)
        });
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.into(),
        backend,
        max_batch_size,
        ..Default::default()
    })
    .expect("runtime");
    eprintln!("backend {:?}", runtime.backend_context().backend_kind);
    let variant = ModelVariant::Lfm3Audio2BGguf;
    let started = Instant::now();
    runtime.load_model(variant).await.expect("load LFM3-Audio");
    let snapshot = runtime.coordinator_snapshot();
    eprintln!(
        "loaded in {:.2}s; reserved {:.2} GiB (unified {:.2} GiB, host {:.2} GiB), process RSS {:.2} GiB",
        started.elapsed().as_secs_f64(),
        snapshot.reserved_memory_bytes as f64 / f64::from(1u32 << 30),
        snapshot.reserved_unified_memory_bytes as f64 / f64::from(1u32 << 30),
        snapshot.reserved_host_memory_bytes as f64 / f64::from(1u32 << 30),
        resident_bytes() as f64 / f64::from(1u32 << 30),
    );
    for model in runtime.core_engine.kv_cache_snapshot().await.models {
        eprintln!(
            "arena: {} tokens per sequence, {} aggregate, {} full-context rows, retained rows {:?}",
            model.single_sequence_token_capacity,
            model.aggregate_token_capacity,
            model.full_context_sequence_capacity,
            model.retained_sequence_capacity
        );
    }
    let model_id = variant.to_string();
    let fox =
        std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav"))
            .expect("fixture");

    // ASR, in English and with the language left to the model.
    for language in [Some("en"), None] {
        let started = Instant::now();
        let transcription = runtime
            .asr_transcribe_bytes(&fox, Some(&model_id), language)
            .await
            .expect("transcribe");
        eprintln!(
            "ASR ({language:?}) -> {:?} in {:.2}s",
            transcription.text,
            started.elapsed().as_secs_f64()
        );
        assert_eq!(words(&transcription.text), words(SENTENCE));
    }
    assert!(runtime
        .asr_transcribe_bytes(&fox, Some(&model_id), Some("tlh"))
        .await
        .is_err());

    // ASR, streamed: the deltas make up the final text.
    let streamed = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = streamed.clone();
    let transcription = runtime
        .asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            &fox,
            Some("en"),
            None,
            None,
            move |delta| sink.lock().unwrap().push_str(&delta),
        )
        .await
        .expect("stream a transcription");
    let streamed = streamed.lock().unwrap().clone();
    eprintln!("ASR stream -> {streamed:?}");
    assert_eq!(streamed.trim(), transcription.text.trim());

    // ASR of 29.7 s in one row, then of 34 s in clips of at most 30 s.
    let (fox_samples, fox_rate) = wav_samples(&fox);
    let test_wav =
        std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/test.wav"))
            .expect("fixture");
    let test_text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/test.md"),
    )
    .expect("fixture");
    let (test_samples, test_rate) = wav_samples(&test_wav);
    assert_eq!(test_rate, fox_rate);
    let pause = vec![0.0; fox_rate as usize];
    let long = [test_samples.as_slice(), &pause, &fox_samples].concat();
    let recall = |reference: &str, heard: &str| {
        let heard = words(heard);
        let reference = words(reference);
        reference.iter().filter(|word| heard.contains(word)).count() as f64 / reference.len() as f64
    };
    for (name, audio, reference) in [
        ("29.7 s", test_wav.clone(), test_text.clone()),
        (
            "34 s",
            wav_bytes(&long, fox_rate),
            format!("{test_text} {SENTENCE}"),
        ),
    ] {
        let started = Instant::now();
        let transcription = runtime
            .asr_transcribe_bytes(&audio, Some(&model_id), Some("en"))
            .await
            .expect("transcribe long audio");
        let recall = recall(&reference, &transcription.text);
        eprintln!(
            "ASR {name} -> recall {recall:.2} in {:.2}s: {:?}",
            started.elapsed().as_secs_f64(),
            transcription.text
        );
        assert!(recall >= 0.9, "{name}: recall {recall:.2}");
    }

    // TTS, final-only.
    let mut request = GenerationRequest::new(SENTENCE);
    request.model_variant = Some(variant);
    request.config.options.voice = Some("male".into());
    request.config.options.speech.seed = Some(5);
    request.language = Some("en-US".into());
    let started = Instant::now();
    let result = runtime.generate(request.clone()).await.expect("speak");
    let seconds = result.samples.len() as f64 / f64::from(result.sample_rate);
    eprintln!(
        "TTS -> {seconds:.2}s of audio in {:.2}s, {} frames, stop {:?}",
        started.elapsed().as_secs_f64(),
        result.total_tokens,
        result.stop_reason
    );
    assert_eq!(result.sample_rate, 24_000);
    assert!((1.5..8.0).contains(&seconds), "{seconds}s");
    let heard = runtime
        .asr_transcribe_bytes(
            &wav_bytes(&result.samples, result.sample_rate),
            Some(&model_id),
            Some("en"),
        )
        .await
        .expect("round trip");
    assert_eq!(words(&heard.text), words(SENTENCE), "{:?}", heard.text);

    // TTS, streamed: the same seed speaks the same audio.
    let (tx, mut rx) = mpsc::channel(256);
    let started = Instant::now();
    let stream = {
        let runtime = &runtime;
        let request = request.clone();
        async move { runtime.generate_streaming(request, tx).await }
    };
    let collect = async {
        let mut samples = Vec::new();
        let mut first = None;
        let mut finals = 0;
        while let Some(chunk) = rx.recv().await {
            if !chunk.samples.is_empty() {
                first.get_or_insert_with(|| started.elapsed());
            }
            samples.extend_from_slice(&chunk.samples);
            finals += usize::from(chunk.is_final);
        }
        (samples, first, finals)
    };
    let (streamed, (samples, first, finals)) = tokio::join!(stream, collect);
    streamed.expect("stream");
    eprintln!(
        "TTS stream: first audio after {:.0} ms, {} samples",
        first.expect("streamed audio").as_secs_f64() * 1000.0,
        samples.len()
    );
    assert_eq!(finals, 1, "one terminal chunk");
    assert_eq!(samples, result.samples, "streamed equals final-only");

    // Concurrent speech batches in the engine: each seeded request speaks
    // exactly what it speaks alone.
    let requests = [
        (SENTENCE, "female", 21),
        ("Batching keeps every voice its own.", "female", 22),
        (SENTENCE, "male", 23),
        ("A late row joins mid-sentence.", "male", 24),
    ]
    .map(|(text, voice, seed)| {
        let mut request = GenerationRequest::new(text);
        request.model_variant = Some(variant);
        request.config.options.voice = Some(voice.into());
        request.config.options.speech.seed = Some(seed);
        request.language = Some("en-US".into());
        request
    });
    let started = Instant::now();
    let mut alone = Vec::new();
    for request in &requests {
        alone.push(
            runtime
                .generate(request.clone())
                .await
                .expect("speak alone"),
        );
    }
    let serial = started.elapsed();
    let before = crate::engine::metrics::engine_batch_metrics_snapshot();
    let started = Instant::now();
    let together = futures::future::join_all(
        requests
            .iter()
            .map(|request| runtime.generate(request.clone())),
    )
    .await;
    let concurrent = started.elapsed();
    let after = crate::engine::metrics::engine_batch_metrics_snapshot();
    let batches = after.model_tensor_batches_total - before.model_tensor_batches_total;
    let rows = after.model_tensor_batch_rows_total - before.model_tensor_batch_rows_total;
    let scalar = after.model_scalar_row_dispatches_total - before.model_scalar_row_dispatches_total;
    eprintln!(
        "TTS x{}: {batches} tensor calls over {rows} rows (mean width {:.2}), {scalar} scalar rows",
        requests.len(),
        rows as f64 / batches.max(1) as f64
    );
    let audio: f64 = alone
        .iter()
        .map(|result| result.samples.len() as f64 / f64::from(result.sample_rate))
        .sum();
    eprintln!(
        "TTS x{}: {audio:.2}s of audio serially in {:.2}s, concurrently in {:.2}s",
        requests.len(),
        serial.as_secs_f64(),
        concurrent.as_secs_f64()
    );
    for (index, (alone, together)) in alone.iter().zip(together).enumerate() {
        let together = together.expect("speak concurrently");
        assert!(!alone.samples.is_empty(), "request {index} spoke");
        assert!(
            alone.samples == together.samples,
            "request {index} differs under concurrency"
        );
    }

    // Concurrent transcriptions batch in the engine: each reads exactly
    // what it reads alone.
    let inputs = [
        fox.clone(),
        wav_bytes(&alone[1].samples, alone[1].sample_rate),
        wav_bytes(&alone[3].samples, alone[3].sample_rate),
        wav_bytes(
            &[vec![0.0; fox_rate as usize], fox_samples.clone()].concat(),
            fox_rate,
        ),
    ];
    let started = Instant::now();
    let mut alone = Vec::new();
    for input in &inputs {
        alone.push(
            runtime
                .asr_transcribe_bytes(input, Some(&model_id), Some("en"))
                .await
                .expect("transcribe alone")
                .text,
        );
    }
    let serial = started.elapsed();
    let before = crate::engine::metrics::engine_batch_metrics_snapshot();
    let started = Instant::now();
    let together = futures::future::join_all(
        inputs
            .iter()
            .map(|input| runtime.asr_transcribe_bytes(input, Some(&model_id), Some("en"))),
    )
    .await;
    let concurrent = started.elapsed();
    let after = crate::engine::metrics::engine_batch_metrics_snapshot();
    let batches = after.model_tensor_batches_total - before.model_tensor_batches_total;
    let rows = after.model_tensor_batch_rows_total - before.model_tensor_batch_rows_total;
    eprintln!(
        "ASR x{}: {batches} tensor calls over {rows} rows (mean width {:.2}); \
         serially {:.2}s, concurrently {:.2}s",
        inputs.len(),
        rows as f64 / batches.max(1) as f64,
        serial.as_secs_f64(),
        concurrent.as_secs_f64()
    );
    for (index, (alone, together)) in alone.iter().zip(together).enumerate() {
        let together = together.expect("transcribe concurrently").text;
        eprintln!("ASR row {index}: {alone:?}");
        assert!(!alone.trim().is_empty(), "row {index} heard speech");
        assert_eq!(*alone, together, "row {index} differs under concurrency");
    }

    // A spoken turn answered with interleaved text and speech.
    let started = Instant::now();
    let reply = runtime
        .speech_to_speech_generate_bytes_with_variant(
            variant,
            &fox,
            Vec::new(),
            GenerationParams {
                max_tokens: 400,
                ..Default::default()
            },
            None,
            None,
        )
        .await
        .expect("voice reply");
    eprintln!(
        "S2S heard {:?}, replied {:?} with {:.2}s of speech in {:.2}s",
        reply.input_transcription,
        reply.text,
        reply.samples.len() as f64 / f64::from(reply.sample_rate),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(
        words(reply.input_transcription.as_deref().unwrap_or_default()),
        words(SENTENCE)
    );
    assert!(!reply.text.trim().is_empty());
    assert!(reply.samples.len() > 24_000 / 2, "the reply speaks");

    // Diarization with LFM3-Audio as its ASR, which transcribes on the
    // long-form graph's invocation workspace.
    let started = Instant::now();
    let diarized = runtime
        .diarize_with_transcript_bytes(
            &test_wav,
            None,
            Some(&model_id),
            None,
            &DiarizationConfig::default(),
            &DiarizationPipelineOptions::default(),
        )
        .await
        .expect("diarize");
    let heard = recall(&test_text, &diarized.asr_text);
    eprintln!(
        "diarized: {} speakers, ASR recall {heard:.2} in {:.2}s",
        diarized.speaker_count,
        started.elapsed().as_secs_f64()
    );
    assert!(heard >= 0.9, "diarization ASR recall {heard:.2}");
}

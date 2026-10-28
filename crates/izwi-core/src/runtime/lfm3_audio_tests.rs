//! LFM3-Audio through the runtime: load, then every public speech route
//! the server calls, on the backend the runtime selects.
//!
//! The test needs the published GGUFs. It runs when
//! `IZWI_LFM3_AUDIO_MODELS_DIR` names the models directory and is otherwise
//! reported as not run.
//!
//! `IZWI_LFM3_AUDIO_BACKEND=metal` runs it on Metal instead of the CPU.
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
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.into(),
        backend,
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
    let started = Instant::now();
    let together = futures::future::join_all(
        requests
            .iter()
            .map(|request| runtime.generate(request.clone())),
    )
    .await;
    let concurrent = started.elapsed();
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
}

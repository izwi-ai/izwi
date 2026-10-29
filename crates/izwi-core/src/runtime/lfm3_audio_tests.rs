//! LFM3-Audio through the runtime: load, then every public speech route
//! the server calls, on the backend the runtime selects.
//!
//! The tests need the published GGUFs. They run when
//! `IZWI_LFM3_AUDIO_MODELS_DIR` names the models directory and is otherwise
//! reported as not run.
//!
//! `IZWI_LFM3_AUDIO_BACKEND=metal` runs them on Metal instead of the CPU,
//! and `IZWI_LFM3_AUDIO_BATCH=4` runs speech and transcription rows four
//! wide.
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

    // Voice chat as rows: a seeded turn replies the same streamed and
    // final-only, and alone and beside other turns.
    let turn = |seed: u64| {
        let mut params = GenerationParams {
            max_tokens: 300,
            ..Default::default()
        };
        params.speech.seed = Some(seed);
        params
    };
    let alone = runtime
        .speech_to_speech_generate_bytes_with_variant(
            variant,
            &fox,
            Vec::new(),
            turn(31),
            None,
            None,
        )
        .await
        .expect("seeded reply");
    let streamed = runtime
        .speech_to_speech_generate_streaming_bytes_with_variant(
            variant,
            &fox,
            Vec::new(),
            turn(31),
            None,
            None,
            |_| {},
        )
        .await
        .expect("streamed reply");
    assert_eq!(streamed.text, alone.text);
    assert!(streamed.samples == alone.samples, "streamed speech differs");
    let turns = [
        (fox.clone(), 31),
        (fox.clone(), 32),
        (inputs[1].clone(), 33),
    ];
    let mut solo = Vec::new();
    for (audio, seed) in &turns {
        solo.push(
            runtime
                .speech_to_speech_generate_bytes_with_variant(
                    variant,
                    audio,
                    Vec::new(),
                    turn(*seed),
                    None,
                    None,
                )
                .await
                .expect("reply alone"),
        );
    }
    let before = crate::engine::metrics::engine_batch_metrics_snapshot();
    let started = Instant::now();
    let together = futures::future::join_all(turns.iter().map(|(audio, seed)| {
        runtime.speech_to_speech_generate_bytes_with_variant(
            variant,
            audio,
            Vec::new(),
            turn(*seed),
            None,
            None,
        )
    }))
    .await;
    let after = crate::engine::metrics::engine_batch_metrics_snapshot();
    let batches = after.model_tensor_batches_total - before.model_tensor_batches_total;
    let rows = after.model_tensor_batch_rows_total - before.model_tensor_batch_rows_total;
    eprintln!(
        "S2S x{}: {batches} tensor calls over {rows} rows (mean width {:.2}) in {:.2}s",
        turns.len(),
        rows as f64 / batches.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    for (index, (solo, together)) in solo.iter().zip(together).enumerate() {
        let together = together.expect("reply concurrently");
        eprintln!(
            "S2S turn {index}: heard {:?}, replied {:?}",
            together.input_transcription, together.text
        );
        assert_eq!(solo.text, together.text, "turn {index} text differs");
        assert!(
            solo.samples == together.samples,
            "turn {index} speech differs: {} vs {} samples, first difference at {:?}",
            solo.samples.len(),
            together.samples.len(),
            solo.samples
                .iter()
                .zip(&together.samples)
                .position(|(a, b)| a != b)
        );
        assert_eq!(solo.input_transcription, together.input_transcription);
    }

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

/// The share of `reference`'s words that `heard` says.
fn recall(reference: &str, heard: &str) -> f64 {
    let heard = words(heard);
    let reference = words(reference);
    reference.iter().filter(|word| heard.contains(word)).count() as f64 / reference.len() as f64
}

/// Word pairs `heard` says more often than `reference` does: a new
/// segment's repeat of the previous one's end shows up here.
fn repeated_pairs(reference: &str, heard: &str) -> Vec<String> {
    let pairs = |text: &str| {
        words(text)
            .windows(2)
            .map(|pair| pair.join(" "))
            .collect::<Vec<_>>()
    };
    let (reference, heard) = (pairs(reference), pairs(heard));
    let count = |pairs: &[String], pair: &String| pairs.iter().filter(|p| *p == pair).count();
    let mut repeated = heard
        .iter()
        .filter(|pair| count(&heard, pair) > count(&reference, pair).max(1))
        .cloned()
        .collect::<Vec<_>>();
    repeated.sort();
    repeated.dedup();
    repeated
}

/// Stream `samples` at `rate` in 100 ms pushes and finish: the transcript
/// the events' deltas make up, each event carrying it so far.
async fn live_transcribe(
    runtime: &RuntimeService,
    model_id: &str,
    samples: &[f32],
    rate: u32,
) -> String {
    let mut stream = runtime
        .try_start_asr_realtime_stream(Some(model_id), Some("en"), None)
        .await
        .expect("start LiveASR")
        .expect("LFM3-Audio streams");
    let mut text = String::new();
    let mut take = |events: Vec<RuntimeAsrRealtimeEvent>| {
        for event in events {
            text.push_str(&event.delta);
            assert_eq!(event.text, text, "an event carries the transcript so far");
        }
    };
    for piece in samples.chunks(rate as usize / 10) {
        take(
            runtime
                .push_asr_realtime_samples(&mut stream, piece, rate)
                .await
                .expect("push"),
        );
    }
    let last = runtime
        .finish_asr_realtime_stream(&mut stream)
        .await
        .expect("finish");
    assert!(last.last().is_some_and(|event| event.is_final));
    take(last);
    assert!(stream.is_closed());
    text
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lfm3_audio_live_asr_streams_on_engine_sessions_if_env_set() {
    let Some(models_dir) = std::env::var_os("IZWI_LFM3_AUDIO_MODELS_DIR") else {
        let _ = writeln!(
            std::io::stderr(),
            "IZWI_LFM3_AUDIO_MODELS_DIR unset; not run"
        );
        return;
    };
    let backend = std::env::var("IZWI_LFM3_AUDIO_BACKEND")
        .ok()
        .and_then(|name| crate::backends::BackendPreference::parse(&name))
        .unwrap_or(crate::backends::BackendPreference::Cpu);
    #[cfg(feature = "metal")]
    let _metal_lease = (backend == crate::backends::BackendPreference::Metal).then(|| {
        crate::backends::metal_test_device_or_skip(
            "lfm3_audio_live_asr_streams_on_engine_sessions_if_env_set",
        )
    });
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.into(),
        backend,
        ..Default::default()
    })
    .expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let variant = ModelVariant::Lfm3Audio2BGguf;
    runtime.load_model(variant).await.expect("load LFM3-Audio");
    let model_id = variant.to_string();
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let (fox, rate) = wav_samples(&std::fs::read(data.join("fox.wav")).expect("fixture"));
    let (speech, speech_rate) =
        wav_samples(&std::fs::read(data.join("test.wav")).expect("fixture"));
    assert_eq!(speech_rate, rate);
    let reference = format!(
        "{} {SENTENCE}",
        std::fs::read_to_string(data.join("test.md")).expect("fixture")
    );
    let long = [speech.as_slice(), &vec![0.0; rate as usize], &fox].concat();

    // LiveASR is English only, and takes no prompt.
    assert!(runtime
        .try_start_asr_realtime_stream(Some(&model_id), Some("fr"), None)
        .await
        .is_err());
    assert!(runtime
        .try_start_asr_realtime_stream(Some(&model_id), None, Some("Ada Lovelace"))
        .await
        .is_err());

    // The 24 kHz pushes are heard at 16 kHz.
    let started = Instant::now();
    let fox_text = live_transcribe(&runtime, &model_id, &fox, rate).await;
    eprintln!(
        "{backend:?}: LiveASR 3.6 s -> {fox_text:?} in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(words(&fox_text), words(SENTENCE));

    // 34 s runs past one segment: the next hears the end again, and the
    // transcript keeps its words once.
    let started = Instant::now();
    let long_text = live_transcribe(&runtime, &model_id, &long, rate).await;
    let heard = recall(&reference, &long_text);
    let seconds = long.len() as f64 / f64::from(rate);
    eprintln!(
        "{backend:?}: LiveASR 34 s -> recall {heard:.2}, RTF {:.2}: {long_text:?}",
        started.elapsed().as_secs_f64() / seconds
    );
    assert!(heard >= 0.9, "LiveASR recall {heard:.2}");
    let repeated = repeated_pairs(&reference, &long_text);
    assert!(repeated.is_empty(), "LiveASR repeated {repeated:?}");

    // Streams at once are engine rows that batch: each writes what it
    // writes alone.
    let started = Instant::now();
    let (long_together, fox_together) = tokio::join!(
        live_transcribe(&runtime, &model_id, &long, rate),
        live_transcribe(&runtime, &model_id, &fox, rate),
    );
    eprintln!(
        "{backend:?}: LiveASR 34 s and 3.6 s at once in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(long_together, long_text);
    assert_eq!(fox_together, fox_text);
}

fn audio_data_url(bytes: &[u8], mime: &str) -> String {
    use base64::Engine as _;
    format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn audio_chat(clips: &[String]) -> crate::models::shared::chat::ChatRequestConfig {
    use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind, ChatRequestConfig};
    ChatRequestConfig {
        media_inputs: clips
            .iter()
            .map(|source| ChatMediaInput {
                kind: ChatMediaKind::Audio,
                source: source.clone(),
            })
            .collect(),
        ..ChatRequestConfig::default()
    }
}

fn user(content: String) -> Vec<crate::models::shared::chat::ChatMessage> {
    vec![crate::models::shared::chat::ChatMessage {
        role: crate::models::shared::chat::ChatRole::User,
        content,
    }]
}

/// LFM3-Audio's greedy answer to a chat question about its clips.
async fn ask_about_audio(
    runtime: &RuntimeService,
    question: String,
    config: crate::models::shared::chat::ChatRequestConfig,
    max_tokens: usize,
) -> crate::runtime::types::ChatGeneration {
    let backend = runtime.backend_context().backend_kind;
    let started = Instant::now();
    let reply = runtime
        .chat_generate_with_runtime_context(
            ModelVariant::Lfm3Audio2BGguf,
            user(question.clone()),
            GenerationParams {
                max_tokens,
                ..greedy_chat()
            },
            config,
            None,
            RuntimeRequestContext::default(),
        )
        .await
        .expect("chat about audio");
    eprintln!(
        "{backend:?}: {question:?} -> {:?} ({:?}, {} prompt + {} tokens) in {:.2}s",
        reply.text,
        reply.finish_reason,
        reply.prompt_tokens,
        reply.tokens_generated,
        started.elapsed().as_secs_f64()
    );
    reply
}

fn greedy_chat() -> GenerationParams {
    GenerationParams {
        temperature: 0.0,
        max_tokens: 96,
        ..GenerationParams::default()
    }
}

/// Chat completions through LFM3-Audio: questions about WAV and MP3 clips,
/// two clips in one turn, streamed and cut-off replies, answered in text;
/// tool calls in chat and in a spoken turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lfm3_audio_answers_chat_about_audio_if_env_set() {
    use crate::models::shared::chat::ChatTruncation;
    use izwi_model_catalog::CHAT_AUDIO_PLACEHOLDER as AUDIO;
    let Some(models_dir) = std::env::var_os("IZWI_LFM3_AUDIO_MODELS_DIR") else {
        let _ = writeln!(
            std::io::stderr(),
            "IZWI_LFM3_AUDIO_MODELS_DIR unset; not run"
        );
        return;
    };
    let backend = std::env::var("IZWI_LFM3_AUDIO_BACKEND")
        .ok()
        .and_then(|name| crate::backends::BackendPreference::parse(&name))
        .unwrap_or(crate::backends::BackendPreference::Cpu);
    #[cfg(feature = "metal")]
    let _metal_lease = (backend == crate::backends::BackendPreference::Metal).then(|| {
        crate::backends::metal_test_device_or_skip("lfm3_audio_answers_chat_about_audio_if_env_set")
    });
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.into(),
        backend,
        ..Default::default()
    })
    .expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let variant = ModelVariant::Lfm3Audio2BGguf;
    runtime.load_model(variant).await.expect("load LFM3-Audio");
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let fox = std::fs::read(data.join("fox.wav")).expect("fixture");
    let speech = std::fs::read(data.join("test.wav")).expect("fixture");
    let greedy = greedy_chat;
    let ask = |question: &str, clips: &[String]| {
        ask_about_audio(&runtime, question.to_string(), audio_chat(clips), 96)
    };

    // A WAV question, answered in text from what the clip says.
    let fox_url = audio_data_url(&fox, "audio/wav");
    let question = format!("{AUDIO}Which animal does the speaker mention? Answer in one word.");
    let reply = ask(&question, std::slice::from_ref(&fox_url)).await;
    assert!(
        reply.text.to_lowercase().contains("fox"),
        "{:?}",
        reply.text
    );
    assert_eq!(reply.finish_reason, Some(izwi_model_catalog::chat::FinishReason::StopToken));
    assert!(reply.tokens_generated > 0 && reply.tokens_generated < 96);
    assert!(reply.prompt_tokens > 20);

    // An MP3 clip (the first 20 s of a longer recording) is heard as its
    // transcription is.
    let mp3 = std::fs::read(data.join("ceo.mp3")).expect("fixture");
    let mp3 = &mp3[..mp3.len().min(200_000)];
    let heard = runtime
        .asr_transcribe_bytes(mp3, Some(&variant.to_string()), Some("en"))
        .await
        .expect("transcribe the MP3")
        .text;
    // The model may think first (its template has no switch for it); the
    // answer follows `</think>`.
    let reply = ask_about_audio(
        &runtime,
        format!("{AUDIO}Repeat exactly what the speaker says."),
        audio_chat(&[audio_data_url(mp3, "audio/mpeg")]),
        512,
    )
    .await;
    assert_eq!(
        reply.finish_reason,
        Some(izwi_model_catalog::chat::FinishReason::StopToken)
    );
    let answer = reply.text.rsplit("</think>").next().unwrap_or_default();
    let repeated = recall(&heard, answer);
    eprintln!("{backend:?}: MP3 heard {heard:?}; repeated recall {repeated:.2}");
    assert!(repeated >= 0.6, "MP3 recall {repeated:.2}");

    // Two clips in one turn, each heard in its place.
    let speech_url = audio_data_url(&speech, "audio/wav");
    let which_mentions_a_fox = format!(
        "First recording: {AUDIO}\nSecond recording: {AUDIO}\n\
         Which recording mentions a fox, the first or the second? Answer in one word."
    );
    let reply = ask(
        &which_mentions_a_fox,
        &[speech_url.clone(), fox_url.clone()],
    )
    .await;
    assert!(
        reply.text.to_lowercase().contains("second"),
        "{:?}",
        reply.text
    );
    let reply = ask(
        &which_mentions_a_fox,
        &[fox_url.clone(), speech_url.clone()],
    )
    .await;
    assert!(
        reply.text.to_lowercase().contains("first"),
        "{:?}",
        reply.text
    );

    // Streamed, the deltas make up the reply (the runtime checks they match
    // its final text exactly).
    let streamed = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = streamed.clone();
    let reply = runtime
        .chat_generate_streaming_tokens_with_runtime_context(
            variant,
            user(format!("{AUDIO}What does the speaker say?")),
            greedy(),
            audio_chat(std::slice::from_ref(&fox_url)),
            None,
            RuntimeRequestContext::default(),
            move |delta, _| sink.lock().unwrap().push_str(&delta),
        )
        .await
        .expect("stream a reply about audio");
    eprintln!("{backend:?}: streamed -> {:?}", reply.text);
    assert_eq!(*streamed.lock().unwrap(), reply.text);
    assert!(recall(SENTENCE, &reply.text) >= 0.8, "{:?}", reply.text);

    // A reply cut off at its budget says so; a budget the context cannot
    // hold is an error unless the request opted into truncation.
    let reply = runtime
        .chat_generate_with_runtime_context(
            variant,
            user(format!("{AUDIO}Describe this recording in detail.")),
            GenerationParams {
                max_tokens: 3,
                ..greedy()
            },
            audio_chat(std::slice::from_ref(&fox_url)),
            None,
            RuntimeRequestContext::default(),
        )
        .await
        .expect("a cut-off reply");
    assert_eq!(reply.finish_reason, Some(izwi_model_catalog::chat::FinishReason::MaxTokens));
    assert_eq!(reply.tokens_generated, 3);
    let mut strict = audio_chat(std::slice::from_ref(&fox_url));
    strict.truncation = ChatTruncation::Disabled;
    let error = runtime
        .chat_generate_with_runtime_context(
            variant,
            user(format!("{AUDIO}Describe this recording.")),
            GenerationParams {
                max_tokens: 10_000_000,
                ..greedy()
            },
            strict,
            None,
            RuntimeRequestContext::default(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::ContextLengthExceeded(_)),
        "{error}"
    );

    // With tools, the reply calls one as LFM3's XML, and answers from its
    // result.
    let mut with_tools = audio_chat(&[]);
    with_tools.tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];
    let question = user("What is the weather in Paris right now?".to_string());
    let tool_budget = || GenerationParams {
        max_tokens: 512,
        ..greedy()
    };
    let call = runtime
        .chat_generate_with_runtime_context(
            variant,
            question.clone(),
            tool_budget(),
            with_tools.clone(),
            None,
            RuntimeRequestContext::default(),
        )
        .await
        .expect("a tool call");
    eprintln!("{backend:?}: tools -> {:?}", call.text);
    assert!(
        call.text
            .contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis"),
        "{:?}",
        call.text
    );
    let mut conversation = question;
    conversation.push(crate::models::shared::chat::ChatMessage {
        role: crate::models::shared::chat::ChatRole::Assistant,
        content: call.text.clone(),
    });
    conversation.push(crate::models::shared::chat::ChatMessage {
        role: crate::models::shared::chat::ChatRole::Tool,
        content: r#"{"temperature_c": 21, "sky": "clear"}"#.to_string(),
    });
    let answer = runtime
        .chat_generate_with_runtime_context(
            variant,
            conversation,
            tool_budget(),
            with_tools,
            None,
            RuntimeRequestContext::default(),
        )
        .await
        .expect("an answer from the tool result");
    eprintln!("{backend:?}: tool result -> {:?}", answer.text);
    let answer = answer.text.rsplit("</think>").next().unwrap_or_default();
    assert!(answer.contains("21") && !answer.contains("<tool_call>"), "{answer:?}");

    // A spoken turn with tools: the first round calls one, the next hears
    // its result and speaks the answer.
    let mut ask = GenerationRequest::new("What time is it right now?");
    ask.model_variant = Some(variant);
    ask.config.options.voice = Some("female".into());
    ask.config.options.speech.seed = Some(3);
    ask.language = Some("en-US".into());
    let spoken = runtime.generate(ask).await.expect("speak the question");
    let question_wav = wav_bytes(&spoken.samples, spoken.sample_rate);
    let time_tool = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "time",
            "description": "Returns the current local date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    })];
    let spoken_params = || GenerationParams {
        max_tokens: 400,
        ..Default::default()
    };
    let first = runtime
        .speech_to_speech_round_streaming_bytes(
            variant,
            &question_wav,
            Vec::new(),
            time_tool.clone(),
            true,
            spoken_params(),
            None,
            None,
            RuntimeRequestContext::default(),
            |_| std::future::ready(Ok(())),
        )
        .await
        .expect("the first round");
    eprintln!(
        "{backend:?}: spoken tools round 0 heard {:?} -> {:?} ({:.2}s of speech)",
        first.input_transcription,
        first.text,
        first.samples.len() as f64 / f64::from(first.sample_rate.max(1))
    );
    assert!(
        first.text.contains("<tool_call>\n<function=time>"),
        "{:?}",
        first.text
    );
    let round = vec![
        crate::models::shared::chat::ChatMessage {
            role: crate::models::shared::chat::ChatRole::User,
            content: AUDIO.to_string(),
        },
        crate::models::shared::chat::ChatMessage {
            role: crate::models::shared::chat::ChatRole::Assistant,
            content: first.text.clone(),
        },
        crate::models::shared::chat::ChatMessage {
            role: crate::models::shared::chat::ChatRole::Tool,
            content: "It is Thursday 29 October 2026, 14:05 (UTC+02:00).".to_string(),
        },
    ];
    let second = runtime
        .speech_to_speech_round_streaming_bytes(
            variant,
            &question_wav,
            round,
            time_tool,
            false,
            spoken_params(),
            None,
            None,
            RuntimeRequestContext::default(),
            |_| std::future::ready(Ok(())),
        )
        .await
        .expect("the answer round");
    let seconds = second.samples.len() as f64 / f64::from(second.sample_rate.max(1));
    eprintln!(
        "{backend:?}: spoken tools round 1 -> {:?} ({seconds:.2}s of speech)",
        second.text
    );
    assert!(second.input_transcription.is_none());
    assert!(seconds > 0.5, "the answer round spoke {seconds:.2}s");

    // A clip the messages do not place is refused before it runs.
    assert!(runtime
        .chat_generate_with_runtime_context(
            variant,
            user("What does the speaker say?".to_string()),
            greedy(),
            audio_chat(std::slice::from_ref(&fox_url)),
            None,
            RuntimeRequestContext::default(),
        )
        .await
        .is_err());
}

/// The edit distance between two token lists over the reference's length.
fn error_rate(reference: &[String], heard: &[String]) -> f64 {
    let mut row = (0..=heard.len()).collect::<Vec<_>>();
    for (i, expected) in reference.iter().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, got) in heard.iter().enumerate() {
            let substitution = previous + usize::from(expected != got);
            previous = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[heard.len()] as f64 / reference.len().max(1) as f64
}

/// Words, or for Japanese (no spaces) its characters, punctuation left out.
fn tokens(language: &str, text: &str) -> Vec<String> {
    if language == "ja" {
        return text
            .chars()
            .filter(|ch| ch.is_alphanumeric())
            .map(String::from)
            .collect();
    }
    words(text)
}

/// LFM3-Audio beyond English, cross-checked by other models: it transcribes
/// French, German, Spanish and Japanese speech Qwen3-TTS 1.7B speaks, and
/// Whisper Large v3 Turbo transcribes the speech it speaks in each. Error
/// rates are word rates, character rates for Japanese.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lfm3_audio_hears_and_speaks_four_languages_if_env_set() {
    let Some(models_dir) = std::env::var_os("IZWI_LFM3_AUDIO_MODELS_DIR") else {
        let _ = writeln!(
            std::io::stderr(),
            "IZWI_LFM3_AUDIO_MODELS_DIR unset; not run"
        );
        return;
    };
    let backend = std::env::var("IZWI_LFM3_AUDIO_BACKEND")
        .ok()
        .and_then(|name| crate::backends::BackendPreference::parse(&name))
        .unwrap_or(crate::backends::BackendPreference::Cpu);
    #[cfg(feature = "metal")]
    let _metal_lease = (backend == crate::backends::BackendPreference::Metal).then(|| {
        crate::backends::metal_test_device_or_skip(
            "lfm3_audio_hears_and_speaks_four_languages_if_env_set",
        )
    });
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.into(),
        backend,
        ..Default::default()
    })
    .expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let lfm3 = ModelVariant::Lfm3Audio2BGguf;
    let speaker = ModelVariant::Qwen3Tts12Hz17BCustomVoice;
    let listener = ModelVariant::WhisperLargeV3Turbo;
    for variant in [lfm3, speaker] {
        runtime.load_model(variant).await.expect("load a model");
    }
    // (language, Qwen3-TTS's name for it, text, highest error rates heard and
    // spoken)
    let cases = [
        (
            "fr",
            "French",
            "Le chat dort sur le canapé pendant que la pluie tombe dehors.",
            // Qwen3-TTS says "canapé" badly (Whisper hears "canneby") and
            // LFM3-Audio then drifts into English for a few words: 0.50 on
            // Metal; it hears its own clean French exactly.
            (0.6, 0.15),
        ),
        (
            "de",
            "German",
            "Der Hund läuft schnell durch den Park und spielt mit dem Ball.",
            (0.15, 0.15),
        ),
        (
            "es",
            "Spanish",
            "Mañana vamos a visitar a mis abuelos en la ciudad de Sevilla.",
            (0.15, 0.15),
        ),
        (
            "ja",
            "Japanese",
            "今日はとても良い天気なので、公園を散歩しましょう。",
            (0.15, 0.15),
        ),
    ];
    // Qwen3-TTS speaks each sentence and LFM3-Audio transcribes it, then
    // LFM3-Audio speaks it.
    let mut heard = Vec::new();
    let mut speech = Vec::new();
    for (language, qwen_language, text, _) in cases {
        let mut request = GenerationRequest::new(text);
        request.model_variant = Some(speaker);
        request.config.options.speaker = Some("Serena".into());
        request.config.options.speech.seed = Some(7);
        request.language = Some(qwen_language.into());
        let spoken = runtime.generate(request).await.expect("Qwen3-TTS speaks");
        let qwen_wav = wav_bytes(&spoken.samples, spoken.sample_rate);
        heard.push(
            runtime
                .asr_transcribe_bytes(&qwen_wav, Some(&lfm3.to_string()), Some(language))
                .await
                .expect("LFM3-Audio transcribes")
                .text,
        );

        let mut request = GenerationRequest::new(text);
        request.model_variant = Some(lfm3);
        request.config.options.voice = Some("female".into());
        request.config.options.speech.seed = Some(7);
        request.language = Some(language.into());
        let spoken = runtime.generate(request).await.expect("LFM3-Audio speaks");
        let lfm3_wav = wav_bytes(&spoken.samples, spoken.sample_rate);
        let self_heard = runtime
            .asr_transcribe_bytes(&lfm3_wav, Some(&lfm3.to_string()), Some(language))
            .await
            .expect("LFM3-Audio transcribes its own speech")
            .text;
        speech.push((qwen_wav, lfm3_wav, self_heard));
    }

    // Whisper transcribes both speakers (the three models do not fit in
    // memory together): Qwen3-TTS's speech checks what LFM3-Audio heard was
    // intelligible.
    for variant in [speaker, lfm3] {
        runtime.unload_model(variant).await.expect("unload a model");
    }
    runtime.load_model(listener).await.expect("load Whisper");
    let mut failures = Vec::new();
    let results = heard.into_iter().zip(speech);
    for ((language, _, text, (max_heard, max_spoken)), (heard, clips)) in
        cases.into_iter().zip(results)
    {
        let (qwen_wav, lfm3_wav, self_heard) = clips;
        let mut whisper = Vec::new();
        for wav in [&qwen_wav, &lfm3_wav] {
            whisper.push(
                runtime
                    .asr_transcribe_bytes(wav, Some(&listener.to_string()), Some(language))
                    .await
                    .expect("Whisper transcribes")
                    .text,
            );
        }
        let reference = tokens(language, text);
        let rate = |transcript: &str| error_rate(&reference, &tokens(language, transcript));
        let (input_rate, heard_rate, spoken_rate, self_rate) = (
            rate(&whisper[0]),
            rate(&heard),
            rate(&whisper[1]),
            rate(&self_heard),
        );
        eprintln!(
            "{backend:?} {language}: input {input_rate:.2} {:?}\n  LFM3-Audio heard \
             {heard_rate:.2} {heard:?}\n  LFM3-Audio spoke {spoken_rate:.2} {:?}\n  \
             and heard itself {self_rate:.2} {self_heard:?}",
            whisper[0], whisper[1]
        );
        // Speech Whisper cannot follow either says nothing about LFM3-Audio
        // (Qwen3-TTS's French varies with the backend's sampling).
        let judged = input_rate <= 0.2;
        if !judged {
            eprintln!("  Qwen3-TTS's {language} input is unclear: LFM3-Audio not judged on it");
        }
        if (judged && heard_rate > max_heard) || spoken_rate > max_spoken || self_rate > 0.15 {
            failures.push(format!(
                "{language}: heard {heard_rate:.2}, spoken {spoken_rate:.2}, \
                 heard itself {self_rate:.2}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

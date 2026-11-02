//! Fish Audio Drama 3 through the runtime on the synthetic checkpoint
//! (`fish_drama3::synthetic`, with the family's stub codec of 2,048
//! samples a frame), on the CPU: load through the real lifecycle,
//! non-streaming and streaming speech, a cloned voice through preparation,
//! requests alone and together, cancellation mid-prefill and mid-decode,
//! the frame budget's error, admission refusal, and unload. The published
//! checkpoint runs the same routes in the e2e kit.

use tokio::sync::mpsc;

use super::*;
use crate::config::EngineConfig;
use crate::model::ModelVariant;

const VARIANT: ModelVariant = ModelVariant::FishAudioDrama3;
/// The stub codec's frame.
const SAMPLES_PER_FRAME: usize = 2_048;
const SAMPLE_RATE: u32 = 44_100;
/// Every request's budget: the synthetic model's random weights draw eos
/// within it for some seeds, and run past it for others.
const BUDGET: usize = 160;

struct ModelsDir(std::path::PathBuf);

impl Drop for ModelsDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn synthetic_models_dir(tag: &str) -> ModelsDir {
    let root = std::env::temp_dir().join(format!(
        "izwi-fish-drama3-runtime-{tag}-{}",
        uuid::Uuid::new_v4()
    ));
    crate::models::architectures::fish_drama3::test_support::write_lifecycle_synthetic_model(
        &root.join(VARIANT.dir_name()),
        5,
    );
    ModelsDir(root)
}

async fn runtime_with(models: &ModelsDir) -> RuntimeService {
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend: crate::backends::BackendPreference::Cpu,
        ..Default::default()
    })
    .expect("runtime");
    runtime
        .load_model(VARIANT)
        .await
        .expect("load the synthetic checkpoint");
    runtime
}

/// A request for `text` with `seed`, its budget [`BUDGET`].
fn request(text: &str, seed: u64) -> GenerationRequest {
    request_within(text, seed, BUDGET)
}

fn request_within(text: &str, seed: u64, budget: usize) -> GenerationRequest {
    let mut request = GenerationRequest::new(text);
    request.model_variant = Some(VARIANT);
    request.config.options.speech.seed = Some(seed);
    request.config.options.max_tokens = budget;
    request
}

/// A two-frame budget: the first frame never draws eos, so a second draw
/// that is not eos runs past it.
const SHORT_BUDGET: usize = 2;

/// `seconds` of a tone at 44.1 kHz as a base64 16-bit WAV: its codes come
/// from the stub codec, a frame per 2,048 samples.
fn reference_wav(seconds: f32) -> String {
    use base64::Engine;
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for i in 0..(seconds * SAMPLE_RATE as f32) as usize {
        let sample = (i as f32 * 0.031).sin() * 0.4;
        writer.write_sample((sample * 32_767.0) as i16).unwrap();
    }
    writer.finalize().unwrap();
    base64::engine::general_purpose::STANDARD.encode(cursor.into_inner())
}

fn with_reference(mut request: GenerationRequest, seconds: f32) -> GenerationRequest {
    request.reference_audio = Some(reference_wav(seconds));
    request.reference_text = Some("hello world.".into());
    request
}

/// Seeds whose generation ends on eos within [`BUDGET`] after at least
/// `min_frames` frames, and one that runs past [`SHORT_BUDGET`], for `text`
/// (the search runs through the runtime itself).
async fn seeds(runtime: &RuntimeService, text: &str, min_frames: usize) -> (Vec<u64>, u64) {
    let mut complete = Vec::new();
    for seed in 0..48 {
        match runtime.generate(request(text, seed)).await {
            Ok(result) if result.samples.len() / SAMPLES_PER_FRAME >= min_frames => {
                complete.push(seed)
            }
            Ok(_) => {}
            Err(error) if error.to_string().contains("frame_limit") => {}
            Err(error) => panic!("seed {seed}: {error}"),
        }
        if complete.len() >= 3 {
            break;
        }
    }
    assert!(complete.len() >= 3, "seeds that end on eos: {complete:?}");
    let mut cut_off = None;
    for seed in 0..16 {
        match runtime
            .generate(request_within(text, seed, SHORT_BUDGET))
            .await
        {
            Ok(_) => {}
            Err(error) if error.to_string().contains("frame_limit") => {
                cut_off = Some(seed);
                break;
            }
            Err(error) => panic!("seed {seed}: {error}"),
        }
    }
    (complete, cut_off.expect("a seed that runs past its budget"))
}

fn assert_whole_frames(samples: &[f32]) -> usize {
    assert!(!samples.is_empty());
    assert_eq!(samples.len() % SAMPLES_PER_FRAME, 0, "{}", samples.len());
    assert!(samples.iter().all(|sample| sample.is_finite()));
    samples.len() / SAMPLES_PER_FRAME
}

fn bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|sample| sample.to_bits()).collect()
}

/// Streams `request` to its end: the chunks' PCM and how many were final.
async fn stream(
    runtime: &RuntimeService,
    request: GenerationRequest,
) -> crate::error::Result<(Vec<f32>, usize)> {
    let (tx, mut rx) = mpsc::channel(256);
    let run = Box::pin(runtime.generate_streaming(request, tx));
    let collect = async {
        let mut samples = Vec::new();
        let mut finals = 0;
        while let Some(chunk) = rx.recv().await {
            if !chunk.samples.is_empty() {
                assert_eq!(chunk.sample_rate, SAMPLE_RATE);
            }
            samples.extend_from_slice(&chunk.samples);
            finals += usize::from(chunk.is_final);
        }
        (samples, finals)
    };
    let (result, collected) = tokio::join!(run, collect);
    result.map(|()| collected)
}

/// The engine holds no request and no retained row.
async fn assert_idle(runtime: &RuntimeService) {
    for _ in 0..200 {
        let coordinator = runtime.coordinator_snapshot();
        let kv = runtime.core_engine.kv_cache_snapshot().await;
        if coordinator.active_jobs == 0
            && coordinator.active_executions == 0
            && kv.totals.registered_sessions == 0
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let coordinator = runtime.coordinator_snapshot();
    let kv = runtime.core_engine.kv_cache_snapshot().await;
    panic!(
        "engine not idle: {} jobs, {} executions, {} retained sessions",
        coordinator.active_jobs, coordinator.active_executions, kv.totals.registered_sessions
    );
}

/// Load, speech with and without streaming, a cloned voice, requests alone
/// and together bit for bit, the budget's error, and unload back to the
/// ledger's baseline.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_drama3_speaks_through_the_runtime_alone_and_together() {
    let _authority = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models = synthetic_models_dir("speech");
    let baseline = RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend: crate::backends::BackendPreference::Cpu,
        ..Default::default()
    })
    .expect("runtime");
    let reserved_before = baseline.coordinator_snapshot().reserved_memory_bytes;
    drop(baseline);
    let runtime = runtime_with(&models).await;
    assert!(runtime.coordinator_snapshot().reserved_memory_bytes > reserved_before);
    assert!(runtime
        .fish_artifact_fingerprint(VARIANT)
        .await
        .is_some_and(|fingerprint| fingerprint.starts_with("synthetic-")));

    let text = "hello world.";
    let (complete, cut_off) = seeds(&runtime, text, 1).await;
    eprintln!("eos seeds {complete:?}, cut-off seed {cut_off}");

    // Final-only: whole frames of PCM, the same bits every time.
    let solo = {
        let mut solo = Vec::new();
        for &seed in &complete[..3] {
            let result = runtime.generate(request(text, seed)).await.expect("speak");
            assert_eq!(result.sample_rate, SAMPLE_RATE);
            let frames = assert_whole_frames(&result.samples);
            assert!((1..=BUDGET).contains(&frames), "{frames}");
            solo.push(result.samples);
        }
        solo
    };
    let again = runtime
        .generate(request(text, complete[0]))
        .await
        .expect("speak again");
    assert_eq!(bits(&again.samples), bits(&solo[0]));

    // Streamed: the same audio in chunks, one final marker.
    let (streamed, finals) = stream(&runtime, request(text, complete[0]))
        .await
        .expect("stream");
    assert_eq!(finals, 1);
    assert_eq!(bits(&streamed), bits(&solo[0]));

    // Together: the same bits as alone, final-only and streamed.
    let together = futures::future::join_all(
        complete[..3]
            .iter()
            .map(|&seed| Box::pin(runtime.generate(request(text, seed)))),
    )
    .await;
    for (result, expected) in together.into_iter().zip(&solo) {
        assert_eq!(bits(&result.expect("together").samples), bits(expected));
    }
    let streams = futures::future::join_all(
        complete[..2]
            .iter()
            .map(|&seed| Box::pin(stream(&runtime, request(text, seed)))),
    )
    .await;
    for (result, expected) in streams.into_iter().zip(&solo) {
        assert_eq!(bits(&result.expect("streamed together").0), bits(expected));
    }

    // A cloned voice through preparation: its reference spans several
    // prefill cells (14 s, about 300 rows), and it speaks the same alone
    // and beside a reference-free request.
    let cloned = with_reference(request(text, complete[1]), 14.0);
    let clone = runtime.generate(cloned.clone()).await;
    let clone = match clone {
        Ok(result) => {
            eprintln!("clone: {} frames", result.samples.len() / SAMPLES_PER_FRAME);
            result.samples
        }
        Err(error) if error.to_string().contains("frame_limit") => {
            // The prompt changed the draws; a cut-off take is still an error.
            Vec::new()
        }
        Err(error) => panic!("clone: {error}"),
    };
    if !clone.is_empty() {
        assert_whole_frames(&clone);
        assert_ne!(bits(&clone), bits(&solo[1]));
        let mut beside = request(text, complete[2]);
        beside.id = "beside-the-clone".into();
        let mut cloned_again = cloned;
        cloned_again.id = "clone-again".into();
        let (clone_again, beside) = tokio::join!(
            Box::pin(runtime.generate(cloned_again)),
            Box::pin(runtime.generate(beside))
        );
        assert_eq!(
            bits(&clone_again.expect("clone again").samples),
            bits(&clone)
        );
        assert_eq!(bits(&beside.expect("beside").samples), bits(&solo[2]));
    }

    // The budget: a take that does not end on eos within it is an error,
    // final-only and streamed, and leaves nothing behind.
    let error = runtime
        .generate(request_within(text, cut_off, SHORT_BUDGET))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("frame_limit"), "{error}");
    let error = stream(&runtime, request_within(text, cut_off, SHORT_BUDGET))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("frame_limit"), "{error}");
    assert_idle(&runtime).await;

    // Unload: the ledger is back where it was before the load.
    runtime.unload_model(VARIANT).await.expect("unload");
    assert_idle(&runtime).await;
    assert!(runtime
        .core_engine
        .kv_cache_snapshot()
        .await
        .models
        .is_empty());
    assert_eq!(
        runtime.coordinator_snapshot().reserved_memory_bytes,
        reserved_before
    );
}

/// Cancellation mid-prefill (a clone whose prompt spans several cells) and
/// mid-decode (a stream cancelled at its first audio): each request ends
/// cancelled, the engine goes idle and the retained row is released, and
/// the next request speaks as it did before.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_drama3_cancels_mid_prefill_and_mid_decode() {
    let _authority = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models = synthetic_models_dir("cancel");
    let runtime = runtime_with(&models).await;
    let text = "hello world.";
    let (complete, _) = seeds(&runtime, text, 24).await;
    let expected = runtime
        .generate(request(text, complete[0]))
        .await
        .expect("speak")
        .samples;

    // Mid-prefill: 28 s of reference (the shared decoder takes 30 s at
    // most) is about 630 rows, three cells. The request is cancelled once
    // its first cell holds a retained row.
    let mut clone = with_reference(request(text, complete[0]), 28.0);
    clone.id = "cancel-mid-prefill".into();
    let id = clone.id.clone();
    let (tx, mut rx) = mpsc::channel::<AudioChunk>(256);
    let audio = tokio::spawn(async move {
        let mut samples = 0;
        while let Some(chunk) = rx.recv().await {
            samples += chunk.samples.len();
        }
        samples
    });
    // The earlier requests' rows are released first, so the next row
    // registered is the clone's.
    assert_idle(&runtime).await;
    let finished = std::sync::atomic::AtomicBool::new(false);
    let run = Box::pin(async {
        let result = runtime.generate_streaming(clone, tx).await;
        finished.store(true, std::sync::atomic::Ordering::SeqCst);
        result
    });
    let cancel = async {
        while !finished.load(std::sync::atomic::Ordering::SeqCst) {
            let registered = runtime
                .core_engine
                .kv_cache_snapshot()
                .await
                .totals
                .registered_sessions;
            if registered > 0 {
                // The abort raises the request's cancellation flag at once;
                // the row may end on it before the core session is reached.
                runtime
                    .core_engine
                    .abort_request(&id)
                    .await
                    .expect("abort mid-prefill");
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        false
    };
    let (result, aborted) = tokio::join!(run, cancel);
    assert!(
        aborted,
        "the clone ended before it was cancelled: {result:?}"
    );
    let samples = audio.await.expect("audio task");
    eprintln!("cancelled mid-prefill: {result:?}");
    let error = result.unwrap_err();
    assert!(
        matches!(error, crate::error::Error::Cancelled(_))
            || error.to_string().to_lowercase().contains("cancel"),
        "{error}"
    );
    assert_eq!(samples, 0, "a cancelled prefill made no audio");
    assert_idle(&runtime).await;

    // Mid-decode: a stream cancelled at its first audio chunk.
    let mut streamed = request(text, complete[0]);
    streamed.id = "cancel-mid-decode".into();
    let id = streamed.id.clone();
    let (tx, mut rx) = mpsc::channel(256);
    let run = Box::pin(runtime.generate_streaming(streamed, tx));
    let cancel_at_first_audio = async {
        let mut samples = 0;
        let mut cancelled = false;
        while let Some(chunk) = rx.recv().await {
            samples += chunk.samples.len();
            if samples > 0 && !cancelled {
                cancelled = true;
                runtime
                    .core_engine
                    .abort_request(&id)
                    .await
                    .expect("abort mid-decode");
            }
        }
        samples
    };
    let (result, samples) = tokio::join!(run, cancel_at_first_audio);
    eprintln!("cancelled mid-decode after {samples} samples");
    let error = result.unwrap_err();
    assert!(
        matches!(error, crate::error::Error::Cancelled(_))
            || error.to_string().to_lowercase().contains("cancel"),
        "{error}"
    );
    assert!(
        samples > 0 && samples < expected.len(),
        "{samples} of {}",
        expected.len()
    );
    assert_idle(&runtime).await;

    // Nothing of the cancelled requests is left: the next speaks as before.
    let again = runtime
        .generate(request(text, complete[0]))
        .await
        .expect("speak after cancellation");
    assert_eq!(bits(&again.samples), bits(&expected));
    runtime.unload_model(VARIANT).await.expect("unload");
}

/// Streams `request` to its end: its PCM and the codes its final chunk
/// carried back.
async fn stream_with_codes(
    runtime: &RuntimeService,
    request: GenerationRequest,
) -> crate::error::Result<(Vec<f32>, Option<crate::runtime::SpeechFrameCodes>)> {
    let (tx, mut rx) = mpsc::channel(256);
    let run = Box::pin(runtime.generate_streaming(request, tx));
    let collect = async {
        let (mut samples, mut codes) = (Vec::new(), None);
        while let Some(chunk) = rx.recv().await {
            samples.extend_from_slice(&chunk.samples);
            if let Some(generated) = chunk.generated_codes {
                assert!(chunk.is_final, "codes come back on the final chunk only");
                codes = Some(std::sync::Arc::unwrap_or_clone(generated));
            }
        }
        (samples, codes)
    };
    let (result, collected) = tokio::join!(run, collect);
    result.map(|()| collected)
}

/// Long text as a chunk session through the runtime (§7.5): each chunk is
/// one request whose prompt holds the earlier chunks it carries (the
/// pinned first and the latest), and whose final chunk hands back the
/// codes it generated, the codes of exactly the audio it streamed. A chunk
/// without its history speaks otherwise. Only a model that continues
/// speech across chunks takes history.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_drama3_chunks_carry_their_history_through_the_runtime() {
    use crate::models::architectures::fish_drama3::test_support::stub_frame;
    use crate::models::architectures::fish_drama3::{
        FishDrama3Codes, FishDrama3History, FishDrama3HistoryTurn,
    };
    use crate::runtime::SpeechChunkTurn;
    let _authority = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models = synthetic_models_dir("chunks");
    let runtime = runtime_with(&models).await;
    let chunks = ["one two.", "three hello.", "world. one."];

    // A seed whose three chunks each end on eos, each chunk carrying the
    // pinned first chunk and the latest, as the server's plan runs them.
    let mut found = None;
    'seeds: for seed in 0..64u64 {
        let mut history: Vec<SpeechChunkTurn> = Vec::new();
        let mut spoken = Vec::new();
        for (index, text) in chunks.iter().enumerate() {
            let mut chunk = request(text, seed + index as u64);
            chunk.chunk_history = history.clone();
            match stream_with_codes(&runtime, chunk).await {
                Ok((samples, Some(codes))) => {
                    let turn = SpeechChunkTurn {
                        text: (*text).into(),
                        codes,
                    };
                    // Pinned first chunk, then the latest.
                    if history.len() == 2 {
                        history.pop();
                    }
                    history.push(turn.clone());
                    spoken.push((samples, turn));
                }
                Ok((_, None)) => panic!("chunk {index} returned no codes"),
                Err(error) if error.to_string().contains("frame_limit") => continue 'seeds,
                Err(error) => panic!("seed {seed} chunk {index}: {error}"),
            }
        }
        found = Some((seed, spoken));
        break;
    }
    let (seed, spoken) = found.expect("a seed whose three chunks end on eos");
    for (samples, turn) in &spoken {
        // The codes are the streamed audio's, frame for frame.
        let frames = assert_whole_frames(samples);
        assert_eq!(turn.codes.frames(), frames);
        assert_eq!(turn.codes.codebooks.len(), 10);
        let decoded = (0..frames)
            .flat_map(|f| {
                let codes = turn
                    .codes
                    .codebooks
                    .iter()
                    .map(|row| row[f])
                    .collect::<Vec<_>>();
                stub_frame(&codes).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(bits(&decoded), bits(samples));
    }

    // The third chunk without its history speaks otherwise: the history
    // reached its prompt.
    let (alone, _) = stream_with_codes(&runtime, request(chunks[2], seed + 2))
        .await
        .unwrap_or_default();
    assert_ne!(bits(&alone), bits(&spoken[2].0));

    // The family's session keeps the same turns the requests carried: each
    // chunk's preparation over the same load holds every one of them.
    let model = runtime
        .model_registry
        .get_fish_drama3_tts_lease(VARIANT)
        .await
        .expect("loaded");
    let mut history = FishDrama3History::new();
    for (index, (text, (_, turn))) in chunks.iter().zip(&spoken).enumerate() {
        let held = history
            .candidates()
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let artifact = model
            .prepare_request(&[], &held, text, BUDGET, &|| Ok(()))
            .expect("prepare in process");
        assert_eq!(artifact.history_turns(), index.min(2), "chunk {index}");
        history.push(FishDrama3HistoryTurn {
            text: (*text).into(),
            codes: FishDrama3Codes::new(turn.codes.codebooks.clone()),
        });
    }

    // A model that does not continue speech across chunks takes no history.
    let mut foreign = GenerationRequest::new("hello");
    foreign.model_variant = Some(ModelVariant::FishAudioS2Pro);
    foreign.chunk_history = vec![spoken[0].1.clone()];
    let error = crate::runtime::request::TtsRuntimeRequest::from_generation(
        foreign,
        ModelVariant::FishAudioS2Pro,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("does not continue speech across chunks"),
        "{error}"
    );
    drop(model);
    runtime.unload_model(VARIANT).await.expect("unload");
}

/// Speaker references through the runtime's preparation: one reference
/// for speaker 0 speaks exactly as the single reference fields; a second
/// reference, and the speaker it voices, change the prompt; references that
/// leave the prompt no room for any text are invalid input naming them,
/// while text too long beside them stays a context error (shorter chunks of
/// it fit).
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn synthetic_drama3_prepares_every_speaker_reference() {
    use crate::models::architectures::fish_drama3::prompt::speaker_reference_transcript;
    use crate::models::architectures::fish_drama3::FishDrama3ReferenceAudio;
    use crate::runtime::SpeechSpeakerReference;
    let _authority = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models = synthetic_models_dir("speakers");
    let runtime = runtime_with(&models).await;
    let (anna, ben) = (reference_wav(1.0), reference_wav(1.5));
    let text = "<|speaker:0|> hello world.\n<|speaker:3|> one two three.";
    let speakers = |references: &[(usize, &str, &str)]| {
        references
            .iter()
            .map(|(speaker, audio, text)| SpeechSpeakerReference {
                speaker: *speaker,
                audio: audio.to_string(),
                text: text.to_string(),
            })
            .collect::<Vec<_>>()
    };
    // A request's outcome: its samples' bits, or its error.
    let outcome = |request: GenerationRequest| {
        let runtime = &runtime;
        async move {
            match stream(runtime, request).await {
                Ok((samples, _)) => Ok(bits(&samples)),
                Err(error) => Err(error.to_string()),
            }
        }
    };
    let mut decided = false;
    for seed in 0..16 {
        let mut single = request(text, seed);
        single.reference_audio = Some(anna.clone());
        single.reference_text = Some("hello world.".into());
        let mut speaker_zero = request(text, seed);
        speaker_zero.speaker_references = speakers(&[(0, &anna, "hello world.")]);
        let mut two = request(text, seed);
        two.speaker_references =
            speakers(&[(0, &anna, "hello world."), (3, &ben, "one two three.")]);
        let mut moved = request(text, seed);
        moved.speaker_references =
            speakers(&[(0, &anna, "hello world."), (5, &ben, "one two three.")]);
        let single = outcome(single).await;
        assert_eq!(outcome(speaker_zero).await, single, "seed {seed}");
        let two = outcome(two).await;
        let moved = outcome(moved).await;
        if let Err(error) = &two {
            assert!(error.contains("frame_limit"), "{error}");
            continue;
        }
        assert_ne!(
            two, single,
            "seed {seed}: the second reference reached the prompt"
        );
        assert_ne!(two, moved, "seed {seed}: its speaker reached the prompt");
        decided = true;
        break;
    }
    assert!(decided, "a seed whose two-reference take ends on eos");

    let model = runtime
        .model_registry
        .get_fish_drama3_tts_lease(VARIANT)
        .await
        .expect("loaded");
    // Four clips of 60 s pass the synthetic model's positions on their own.
    let clip = (0..60 * 8_000)
        .map(|i| (i as f32 * 0.2).sin() * 0.4)
        .collect::<Vec<_>>();
    let transcripts = (0..4)
        .map(|speaker| speaker_reference_transcript(speaker, "hello world.").unwrap())
        .collect::<Vec<_>>();
    let long_references = transcripts
        .iter()
        .map(|transcript| {
            (
                transcript.as_str(),
                FishDrama3ReferenceAudio {
                    samples: &clip,
                    channels: 1,
                    sample_rate: 8_000,
                },
            )
        })
        .collect::<Vec<_>>();
    let error = model
        .prepare_request(&long_references, &[], "hello world.", BUDGET, &|| Ok(()))
        .expect_err("no room for the text");
    assert!(
        matches!(&error, crate::error::Error::InvalidInput(message)
            if message.contains("holds 4 references in") && message.contains("no room")),
        "{error}"
    );
    // One short clip beside text too long for the prompt: a context error
    // that names the reference's share.
    let long_text = "one two three hello world. ".repeat(1_000);
    let error = model
        .prepare_request(&long_references[..1], &[], &long_text, BUDGET, &|| Ok(()))
        .expect_err("text too long");
    assert!(
        matches!(&error, crate::error::Error::ContextLengthExceeded(message)
            if message.contains("it holds 1 reference in")),
        "{error}"
    );
    drop(model);
    runtime.unload_model(VARIANT).await.expect("unload");
}

/// A machine that cannot hold the load is refused at admission, before the
/// model is published; once the memory is free the same load succeeds.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn synthetic_drama3_load_is_refused_when_the_device_is_too_small() {
    use crate::engine::{ReservationClass, ReservationOwner, ResourceAmount, ResourceVector};
    let _authority = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models = synthetic_models_dir("admission");
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models.0.clone(),
        backend: crate::backends::BackendPreference::Cpu,
        ..Default::default()
    })
    .expect("runtime");
    let authority = runtime.coordinator.resource_authority();
    let headroom = match authority
        .planning_headroom_bytes(crate::backends::BackendKind::Cpu)
        .expect("headroom")
    {
        ResourceAmount::Known(bytes) => bytes,
        ResourceAmount::Unknown => {
            eprintln!("the CPU authority reports no headroom; not run");
            return;
        }
    };
    // Everything but 64 MiB is taken: less than the codec alone.
    let hog = authority
        .reserve(
            ReservationOwner::new(ReservationClass::Model, "drama3-admission-test"),
            ResourceVector {
                host_bytes: ResourceAmount::Known(headroom.saturating_sub(64 << 20)),
                ..ResourceVector::zero()
            },
        )
        .expect("hold the headroom");
    let error = runtime.load_model(VARIANT).await.unwrap_err().to_string();
    assert!(error.contains("Cannot load Fish Audio Drama 3"), "{error}");
    assert!(runtime
        .model_registry
        .get_fish_drama3_tts(VARIANT)
        .await
        .is_none());
    assert!(runtime
        .core_engine
        .kv_cache_snapshot()
        .await
        .models
        .is_empty());
    drop(hog);
    runtime
        .load_model(VARIANT)
        .await
        .expect("load once the memory is free");
    assert!(runtime
        .model_registry
        .get_fish_drama3_tts(VARIANT)
        .await
        .is_some());
    runtime.unload_model(VARIANT).await.expect("unload");
}

/// What the Metal and CUDA runtime tests share.
#[cfg(any(feature = "metal", feature = "cuda"))]
mod device_support {
    use super::*;

    /// Sets a variable for the test (under `env_test_lock`) and restores
    /// it when dropped.
    #[cfg(feature = "metal")]
    pub(super) struct EnvValue(&'static str, Option<std::ffi::OsString>);

    #[cfg(feature = "metal")]
    impl EnvValue {
        pub(super) fn set(name: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var_os(name);
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
            Self(name, previous)
        }
    }

    #[cfg(feature = "metal")]
    impl Drop for EnvValue {
        fn drop(&mut self) {
            match self.1.take() {
                Some(value) => std::env::set_var(self.0, value),
                None => std::env::remove_var(self.0),
            }
        }
    }

    /// Cancels `request` (streamed) once `ready` says so; returns the result
    /// and the PCM samples it made.
    pub(super) async fn cancel_streamed(
        runtime: &RuntimeService,
        request: GenerationRequest,
        at_first_audio: bool,
    ) -> (crate::error::Result<()>, usize) {
        let id = request.id.clone();
        let (tx, mut rx) = mpsc::channel::<AudioChunk>(256);
        let finished = std::sync::atomic::AtomicBool::new(false);
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        let run = Box::pin(async {
            let result = runtime.generate_streaming(request, tx).await;
            finished.store(true, std::sync::atomic::Ordering::SeqCst);
            result
        });
        let watch = async {
            let mut samples = 0;
            loop {
                if !at_first_audio
                    && !cancelled.load(std::sync::atomic::Ordering::SeqCst)
                    && runtime
                        .core_engine
                        .kv_cache_snapshot()
                        .await
                        .totals
                        .registered_sessions
                        > 0
                {
                    cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
                    runtime
                        .core_engine
                        .abort_request(&id)
                        .await
                        .expect("abort mid-prefill");
                }
                match tokio::time::timeout(std::time::Duration::from_millis(1), rx.recv()).await {
                    Ok(Some(chunk)) => {
                        samples += chunk.samples.len();
                        if at_first_audio
                            && samples > 0
                            && !cancelled.swap(true, std::sync::atomic::Ordering::SeqCst)
                        {
                            runtime
                                .core_engine
                                .abort_request(&id)
                                .await
                                .expect("abort mid-decode");
                        }
                    }
                    Ok(None) => return samples,
                    Err(_) if finished.load(std::sync::atomic::Ordering::SeqCst) => return samples,
                    Err(_) => {}
                }
            }
        };
        let (result, samples) = tokio::join!(run, watch);
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "the request ended before it was cancelled: {result:?}"
        );
        (result, samples)
    }

    pub(super) fn assert_cancelled(result: crate::error::Result<()>) {
        let error = result.unwrap_err();
        assert!(
            matches!(error, crate::error::Error::Cancelled(_))
                || error.to_string().to_lowercase().contains("cancel"),
            "{error}"
        );
    }
}

/// The same routes on Metal (the synthetic checkpoint with the published
/// 128-dim DeltaNet heads, so the fused kernels run): load, speech alone,
/// streamed and together bit for bit, the budget's error, cancellation
/// mid-prefill and mid-decode, unload back to the ledger's baseline; at
/// width 1 and at width 2 (`IZWI_FISH_DRAMA3_NATIVE_BATCHING`), where
/// concurrent requests share decode steps.
#[cfg(feature = "metal")]
mod metal {
    use super::device_support::{assert_cancelled, cancel_streamed, EnvValue};
    use super::*;
    use crate::models::architectures::fish_drama3::synthetic::SyntheticGeometry;

    fn models_dir(tag: &str) -> ModelsDir {
        let root = std::env::temp_dir().join(format!(
            "izwi-fish-drama3-metal-runtime-{tag}-{}",
            uuid::Uuid::new_v4()
        ));
        crate::models::architectures::fish_drama3::test_support::write_lifecycle_synthetic_model_in(
            &root.join(VARIANT.dir_name()),
            5,
            SyntheticGeometry::KernelHeads,
        );
        ModelsDir(root)
    }

    fn config(models: &ModelsDir) -> EngineConfig {
        EngineConfig {
            models_dir: models.0.clone(),
            backend: crate::backends::BackendPreference::Metal,
            ..Default::default()
        }
    }

    #[allow(clippy::await_holding_lock)]
    async fn speaks_alone_and_together(width_two: bool, test: &str) {
        let Some((_device, _lease)) = crate::backends::metal_test_device_or_skip(test) else {
            return;
        };
        let _authority = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _batching = EnvValue::set(
            crate::runtime::fish_drama3_capacity::NATIVE_BATCHING_ENV,
            Some(if width_two { "1" } else { "0" }),
        );
        // A row's planned codec workspace and the two reference preparations
        // are the published codec's (about 2 GB and 23 GB), so a 16 GB Mac's
        // default budget seals one row. Width 2 raises the advisory budget so
        // two rows seal; the synthetic rows allocate a few MB each.
        let _budget = EnvValue::set(
            "IZWI_METAL_MEMORY_BUDGET_BYTES",
            width_two.then_some("51539607552"),
        );
        let models = models_dir(if width_two { "width-2" } else { "width-1" });
        let baseline = RuntimeService::new(config(&models)).expect("runtime");
        let reserved_before = baseline.coordinator_snapshot().reserved_memory_bytes;
        drop(baseline);
        let runtime = RuntimeService::new(config(&models)).expect("runtime");
        runtime
            .load_model(VARIANT)
            .await
            .expect("load the synthetic checkpoint on Metal");
        assert!(runtime.coordinator_snapshot().reserved_memory_bytes > reserved_before);
        let model = runtime
            .model_registry
            .get_fish_drama3_tts(VARIANT)
            .await
            .expect("the loaded model");
        assert!(model.device().is_metal());
        let checks = model.metal_checks().expect("the Metal checks ran");
        assert_eq!(checks.f16_range_breaches, 0);

        let text = "hello world.";
        let (complete, cut_off) = seeds(&runtime, text, 2).await;
        eprintln!("Metal eos seeds {complete:?}, cut-off seed {cut_off}");
        let mut solo = Vec::new();
        for &seed in &complete[..3] {
            let result = runtime.generate(request(text, seed)).await.expect("speak");
            assert_whole_frames(&result.samples);
            solo.push(result.samples);
        }
        let again = runtime
            .generate(request(text, complete[0]))
            .await
            .expect("speak again");
        assert_eq!(bits(&again.samples), bits(&solo[0]));
        let (streamed, finals) = stream(&runtime, request(text, complete[0]))
            .await
            .expect("stream");
        assert_eq!(finals, 1);
        assert_eq!(bits(&streamed), bits(&solo[0]));
        let multirow = || {
            crate::engine::metrics::engine_batch_metrics_snapshot()
                .model_tensor_multirow_calls_total
        };
        let shared_steps = multirow();
        let together = futures::future::join_all(
            complete[..3]
                .iter()
                .map(|&seed| Box::pin(runtime.generate(request(text, seed)))),
        )
        .await;
        for (result, expected) in together.into_iter().zip(&solo) {
            assert_eq!(bits(&result.expect("together").samples), bits(expected));
        }
        if width_two {
            // Width 2: concurrent rows shared decode steps (the counter is
            // process-wide, so width 1 cannot assert its absence).
            assert!(multirow() > shared_steps, "no decode step ran two rows");
        }
        let streams = futures::future::join_all(
            complete[..2]
                .iter()
                .map(|&seed| Box::pin(stream(&runtime, request(text, seed)))),
        )
        .await;
        for (result, expected) in streams.into_iter().zip(&solo) {
            assert_eq!(bits(&result.expect("streamed together").0), bits(expected));
        }
        let error = runtime
            .generate(request_within(text, cut_off, SHORT_BUDGET))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("frame_limit"), "{error}");
        assert_idle(&runtime).await;

        // Cancellation: mid-prefill (a clone over three cells) and at the
        // first audio of a stream; the next request speaks as before.
        let mut clone = with_reference(request(text, complete[0]), 28.0);
        clone.id = "metal-cancel-mid-prefill".into();
        let (result, samples) = cancel_streamed(&runtime, clone, false).await;
        assert_cancelled(result);
        assert_eq!(samples, 0, "a cancelled prefill made no audio");
        assert_idle(&runtime).await;
        let mut streamed = request(text, complete[0]);
        streamed.id = "metal-cancel-mid-decode".into();
        let (result, samples) = cancel_streamed(&runtime, streamed, true).await;
        assert_cancelled(result);
        assert!(samples > 0 && samples < solo[0].len(), "{samples}");
        assert_idle(&runtime).await;
        let after = runtime
            .generate(request(text, complete[0]))
            .await
            .expect("speak after cancellation");
        assert_eq!(bits(&after.samples), bits(&solo[0]));

        // Every frame sampled on the device.
        let summary = model.diagnostics_summary();
        assert!(
            summary["resident_sampling"]["frames_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        drop(model);
        runtime.unload_model(VARIANT).await.expect("unload");
        assert_idle(&runtime).await;
        assert!(runtime
            .core_engine
            .kv_cache_snapshot()
            .await
            .models
            .is_empty());
        assert_eq!(
            runtime.coordinator_snapshot().reserved_memory_bytes,
            reserved_before
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn synthetic_drama3_speaks_on_metal_at_width_1() {
        speaks_alone_and_together(false, "synthetic_drama3_speaks_on_metal_at_width_1").await;
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn synthetic_drama3_speaks_on_metal_at_width_2() {
        speaks_alone_and_together(true, "synthetic_drama3_speaks_on_metal_at_width_2").await;
    }
}

/// The same routes on CUDA (the synthetic checkpoint with the published
/// 128-dim DeltaNet heads): load through the SM80 floor and the CUDA
/// checks, speech alone, streamed and together bit for bit, the budget's
/// error, cancellation mid-prefill and mid-decode, unload back to the
/// ledger's baseline; at width 1 (`max_batch_size` fixed at one) and at the
/// automatic width 8, where eight concurrent requests share decode steps.
/// They skip without a CUDA device and have only been compiled so far.
#[cfg(feature = "cuda")]
mod cuda {
    use super::device_support::{assert_cancelled, cancel_streamed};
    use super::*;
    use crate::models::architectures::fish_drama3::synthetic::SyntheticGeometry;

    fn models_dir(tag: &str) -> ModelsDir {
        let root = std::env::temp_dir().join(format!(
            "izwi-fish-drama3-cuda-runtime-{tag}-{}",
            uuid::Uuid::new_v4()
        ));
        crate::models::architectures::fish_drama3::test_support::write_lifecycle_synthetic_model_in(
            &root.join(VARIANT.dir_name()),
            5,
            SyntheticGeometry::KernelHeads,
        );
        ModelsDir(root)
    }

    fn config(models: &ModelsDir, width: Option<usize>) -> EngineConfig {
        EngineConfig {
            models_dir: models.0.clone(),
            backend: crate::backends::BackendPreference::Cuda,
            max_batch_size: width.map_or(crate::config::BatchSizePreference::Auto, |rows| {
                crate::config::BatchSizePreference::fixed(rows).expect("a positive width")
            }),
            ..Default::default()
        }
    }

    #[allow(clippy::await_holding_lock)]
    async fn speaks_alone_and_together(width: usize) {
        let Some(_device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let _authority = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let fixed = (width == 1).then_some(1);
        let models = models_dir(&format!("width-{width}"));
        let baseline = RuntimeService::new(config(&models, fixed)).expect("runtime");
        let reserved_before = baseline.coordinator_snapshot().reserved_memory_bytes;
        drop(baseline);
        let runtime = RuntimeService::new(config(&models, fixed)).expect("runtime");
        runtime
            .load_model(VARIANT)
            .await
            .expect("load the synthetic checkpoint on CUDA");
        assert!(runtime.coordinator_snapshot().reserved_memory_bytes > reserved_before);
        let model = runtime
            .model_registry
            .get_fish_drama3_tts(VARIANT)
            .await
            .expect("the loaded model");
        assert!(model.device().is_cuda());
        let checks = model.cuda_checks().expect("the CUDA checks ran");
        assert_eq!(checks.row_gemv, "verified");
        assert_eq!(model.kv_dtype(), candle_core::DType::BF16);

        let text = "hello world.";
        let (complete, cut_off) = seeds(&runtime, text, 2).await;
        eprintln!("CUDA eos seeds {complete:?}, cut-off seed {cut_off}");
        let mut solo = Vec::new();
        for &seed in &complete[..3] {
            let result = runtime.generate(request(text, seed)).await.expect("speak");
            assert_whole_frames(&result.samples);
            solo.push(result.samples);
        }
        let again = runtime
            .generate(request(text, complete[0]))
            .await
            .expect("speak again");
        assert_eq!(bits(&again.samples), bits(&solo[0]));
        let (streamed, finals) = stream(&runtime, request(text, complete[0]))
            .await
            .expect("stream");
        assert_eq!(finals, 1);
        assert_eq!(bits(&streamed), bits(&solo[0]));
        let multirow = || {
            crate::engine::metrics::engine_batch_metrics_snapshot()
                .model_tensor_multirow_calls_total
        };
        let shared_steps = multirow();
        // Eight requests at once (the three seeds, repeated): each must make
        // its solo bits whatever rows share its steps.
        let together = futures::future::join_all(
            (0..8).map(|i| Box::pin(runtime.generate(request(text, complete[i % 3])))),
        )
        .await;
        for (i, result) in together.into_iter().enumerate() {
            assert_eq!(
                bits(&result.expect("together").samples),
                bits(&solo[i % 3]),
                "request {i}"
            );
        }
        if width > 1 {
            assert!(multirow() > shared_steps, "no decode step ran several rows");
        }
        let streams = futures::future::join_all(
            complete[..3]
                .iter()
                .map(|&seed| Box::pin(stream(&runtime, request(text, seed)))),
        )
        .await;
        for (result, expected) in streams.into_iter().zip(&solo) {
            assert_eq!(bits(&result.expect("streamed together").0), bits(expected));
        }
        let error = runtime
            .generate(request_within(text, cut_off, SHORT_BUDGET))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("frame_limit"), "{error}");
        assert_idle(&runtime).await;

        let mut clone = with_reference(request(text, complete[0]), 28.0);
        clone.id = "cuda-cancel-mid-prefill".into();
        let (result, samples) = cancel_streamed(&runtime, clone, false).await;
        assert_cancelled(result);
        assert_eq!(samples, 0, "a cancelled prefill made no audio");
        assert_idle(&runtime).await;
        let mut streamed = request(text, complete[0]);
        streamed.id = "cuda-cancel-mid-decode".into();
        let (result, samples) = cancel_streamed(&runtime, streamed, true).await;
        assert_cancelled(result);
        assert!(samples > 0 && samples < solo[0].len(), "{samples}");
        assert_idle(&runtime).await;
        let after = runtime
            .generate(request(text, complete[0]))
            .await
            .expect("speak after cancellation");
        assert_eq!(bits(&after.samples), bits(&solo[0]));

        let summary = model.diagnostics_summary();
        assert!(
            summary["resident_sampling"]["frames_total"]
                .as_u64()
                .unwrap()
                > 0
        );
        drop(model);
        runtime.unload_model(VARIANT).await.expect("unload");
        assert_idle(&runtime).await;
        assert!(runtime
            .core_engine
            .kv_cache_snapshot()
            .await
            .models
            .is_empty());
        assert_eq!(
            runtime.coordinator_snapshot().reserved_memory_bytes,
            reserved_before
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn synthetic_drama3_speaks_on_cuda_at_width_1() {
        speaks_alone_and_together(1).await;
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn synthetic_drama3_speaks_on_cuda_at_width_8() {
        speaks_alone_and_together(8).await;
    }
}

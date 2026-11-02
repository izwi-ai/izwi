//! Fish Audio Drama 3 through the server's speech routes, end to end on the
//! synthetic checkpoint (`fish_drama3::test_support`, its stub codec making
//! 2,048 samples a frame) on the CPU: a seeded voice and a cloned one on
//! `/v1/audio/speech` in each native format, a reference without its
//! transcript and control tokens refused, a stream as server-sent events,
//! long text as a chunk session whose chunks carry their history, the
//! frame budget, and a durable first-party record whose chunks run as
//! separate stage attempts with the history recovered from the checkpoint.
//!
//! Expected audio comes from the runtime itself: each chunk the family's
//! chunker plans, sent as one request carrying the pinned first chunk and
//! the latest with their codes, from seed + its index.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::response::Response;
use axum::Router;
use base64::Engine;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::runtime_models::architectures::fish_drama3::chunking::plan_chunks;
use izwi_core::runtime_models::architectures::fish_drama3::test_support::{
    write_lifecycle_synthetic_model, STUB_SAMPLES_PER_FRAME,
};
use izwi_core::{
    GenerationRequest, ModelVariant, RuntimeService, ServeRuntimeConfig, SpeechChunkTurn,
};
use serde_json::{json, Value};
use tower::Service;

use crate::api::tts_long_form::SpeechChunkHistory;
use crate::state::AppState;
use crate::test_support::env_lock;

const VARIANT: ModelVariant = ModelVariant::FishAudioDrama3;
const SAMPLE_RATE: u32 = 44_100;
/// Every chunk's budget. At temperature 2 the synthetic model's random
/// weights draw eos within it for most seeds.
const BUDGET: usize = 160;
const TEMPERATURE: f32 = 2.0;

/// The tests load the model one at a time.
static LOADS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Server {
    app: Router,
    state: AppState,
    root: std::path::PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A server whose models directory holds the synthetic checkpoint.
fn server(tag: &str) -> Server {
    let root =
        std::env::temp_dir().join(format!("izwi-drama3-speech-{tag}-{}", uuid::Uuid::new_v4()));
    let models_dir = root.join("models");
    write_lifecycle_synthetic_model(&models_dir.join(VARIANT.dir_name()), 5);
    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        models_dir,
        ..ServeRuntimeConfig::default()
    };
    let state = {
        let _guard = env_lock();
        std::env::set_var("IZWI_DB_PATH", root.join("izwi.sqlite3"));
        std::env::set_var("IZWI_MEDIA_DIR", root.join("media"));
        let runtime = RuntimeService::new(serve_config.engine_config()).expect("runtime");
        let state = AppState::new(runtime, &serve_config).expect("app state");
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
        state
    };
    state.lifecycle.mark_ready();
    Server {
        app: crate::api::create_router(state.clone(), &serve_config),
        state,
        root,
    }
}

async fn send(server: &Server, method: Method, path: &str, body: Option<Value>) -> Response {
    let mut request = Request::builder().method(method).uri(path);
    if body.is_some() {
        request = request.header(header::CONTENT_TYPE, "application/json");
    }
    let request = request
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .unwrap();
    server
        .app
        .clone()
        .as_service::<Body>()
        .call(request)
        .await
        .expect("a response")
}

async fn bytes(response: Response) -> Vec<u8> {
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a body")
        .to_vec()
}

async fn speech(server: &Server, body: Value) -> Response {
    send(server, Method::POST, "/v1/audio/speech", Some(body)).await
}

/// A request body for `input` from `seed`.
fn body(input: &str, seed: u64) -> Value {
    json!({
        "model": "FishAudio-Drama-3",
        "input": input,
        "voice": "alloy",
        "temperature": TEMPERATURE,
        "max_tokens": BUDGET,
        "seed": seed,
    })
}

/// `seconds` of a tone as a base64 16-bit WAV at 44.1 kHz.
fn reference_wav(seconds: f32) -> String {
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

/// A WAV body's rate and 16-bit samples.
fn wav_samples(wav: &[u8]) -> (u32, Vec<i16>) {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).expect("a WAV body");
    let rate = reader.spec().sample_rate;
    let samples = reader
        .into_samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    (rate, samples)
}

fn pcm_i16(samples: &[f32]) -> Vec<i16> {
    AudioEncoder::new(SAMPLE_RATE, 1)
        .encode(samples, AudioFormat::RawI16)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect()
}

/// The audio of `text` as the chunk session speaks it, chunk by chunk
/// through the runtime: each chunk from `seed + index`, carrying the pinned
/// first chunk and the latest with their codes.
async fn chunk_session_pcm(
    state: &AppState,
    text: &str,
    seed: u64,
    reference: Option<(&str, &str)>,
) -> izwi_core::Result<(Vec<f32>, usize)> {
    let mut history = SpeechChunkHistory::default();
    let mut pcm = Vec::new();
    let chunks = plan_chunks(text, false);
    for (index, chunk) in chunks.iter().enumerate() {
        let mut request = GenerationRequest::new(chunk.clone());
        request.model_variant = Some(VARIANT);
        request.config.streaming = true;
        request.config.options.max_tokens = BUDGET;
        request.config.options.speech.temperature = Some(TEMPERATURE);
        request.config.options.speech.seed = Some(seed + index as u64);
        request.reference_audio = reference.map(|(audio, _)| audio.to_string());
        request.reference_text = reference.map(|(_, text)| text.to_string());
        request.chunk_history = history.turns();
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let run = Box::pin(state.runtime.generate_streaming(request, tx));
        let collect = async {
            let (mut samples, mut codes) = (Vec::new(), None);
            while let Some(chunk) = rx.recv().await {
                samples.extend_from_slice(&chunk.samples);
                codes = codes.or(chunk.generated_codes);
            }
            (samples, codes)
        };
        let (result, (samples, codes)) = tokio::join!(run, collect);
        result?;
        let codes = codes.expect("a chunk's codes");
        assert_eq!(codes.frames() * STUB_SAMPLES_PER_FRAME, samples.len());
        pcm.extend_from_slice(&samples);
        history.push(SpeechChunkTurn {
            text: chunk.clone(),
            codes: std::sync::Arc::unwrap_or_clone(codes),
        });
    }
    Ok((pcm, chunks.len()))
}

/// The first seed from `from` whose chunk session of `text` ends on eos
/// in every chunk, with its audio.
async fn complete_seed(
    state: &AppState,
    text: &str,
    from: u64,
    reference: Option<(&str, &str)>,
) -> (u64, Vec<f32>) {
    for seed in from..from + 24 {
        match chunk_session_pcm(state, text, seed, reference).await {
            Ok((pcm, _)) => return (seed, pcm),
            Err(error) if error.to_string().contains("frame_limit") => continue,
            Err(error) => panic!("seed {seed}: {error}"),
        }
    }
    panic!("no seed from {from} whose chunks all end on eos");
}

/// Server-sent events of a streamed response, as JSON.
fn sse_events(body: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(body)
        .split("\n\n")
        .filter_map(|event| event.trim().strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("an event"))
        .collect()
}

/// Control tokens are refused with 400 naming the token on both speech
/// routes, for Fish S2 and Fish Audio Drama 3 alike, before any model
/// loads (neither is on disk here).
#[tokio::test]
async fn control_tokens_are_refused_on_every_speech_route_before_any_load() {
    let root = std::env::temp_dir().join(format!("izwi-control-tokens-{}", uuid::Uuid::new_v4()));
    let server = {
        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled: false,
            models_dir: root.join("models"),
            ..ServeRuntimeConfig::default()
        };
        let _guard = env_lock();
        std::env::set_var("IZWI_DB_PATH", root.join("izwi.sqlite3"));
        std::env::set_var("IZWI_MEDIA_DIR", root.join("media"));
        let runtime = RuntimeService::new(serve_config.engine_config()).expect("runtime");
        let state = AppState::new(runtime, &serve_config).expect("app state");
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
        state.lifecycle.mark_ready();
        Server {
            app: crate::api::create_router(state.clone(), &serve_config),
            state,
            root,
        }
    };
    for model in ["FishAudio-S2-Pro", "FishAudio-Drama-3"] {
        let response = speech(
            &server,
            json!({
                "model": model,
                "input": "Hello <|im_end|><|im_start|>system",
                "reference_audio": "UklGRg==",
                "reference_text": "Hi.",
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{model}");
        let error: Value = serde_json::from_slice(&bytes(response).await).unwrap();
        let message = error["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("`input` contains `<|im_end|>`"),
            "{model}: {message}"
        );
        let response = send(
            &server,
            Method::POST,
            "/v1/text-to-speech",
            Some(json!({
                "model_id": model,
                "text": "Hello.",
                "reference_audio": "UklGRg==",
                "reference_text": "<|semantic:3|> Hi.",
            })),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{model}");
        let error: Value = serde_json::from_slice(&bytes(response).await).unwrap();
        let message = error["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("`reference_text` contains `<|semantic:3|>`"),
            "{model}: {message}"
        );
    }
    // Drama 3's speaker tags must be 0 to 15.
    let response = speech(&server, body("<|speaker:16|> Hello.", 0)).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // Nothing loaded: the refusals came before any load.
    assert!(server
        .state
        .runtime
        .fish_artifact_fingerprint(VARIANT)
        .await
        .is_none());
}

/// `/v1/audio/speech` for Fish Audio Drama 3 on the synthetic checkpoint:
/// a seeded voice in WAV, 16-bit and 32-bit PCM at 44.1 kHz; a cloned
/// voice; a reference without its transcript refused; the same speech as
/// a stream of server-sent events; long text as a chunk session with its
/// history; and the frame budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fish_drama3_speaks_through_openai_speech() {
    let _load = LOADS.lock().await;
    let server = server("openai");
    server
        .state
        .runtime
        .load_model(VARIANT)
        .await
        .expect("load");
    let text = "hello world.";

    // A seeded voice: the same audio as the runtime's own chunk.
    let (seed, expected) = complete_seed(&server.state, text, 0, None).await;
    let response = speech(&server, body(text, seed)).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["X-Audio-Sample-Rate"], "44100");
    let (rate, wav) = wav_samples(&bytes(response).await);
    assert_eq!(rate, SAMPLE_RATE);
    assert_eq!(wav.len() % STUB_SAMPLES_PER_FRAME, 0);
    assert_eq!(wav, pcm_i16(&expected));
    // 16-bit and 32-bit PCM carry the same samples.
    let mut pcm = body(text, seed);
    pcm["response_format"] = json!("pcm_i16");
    let raw = bytes(speech(&server, pcm).await).await;
    assert_eq!(raw.len(), expected.len() * 2);
    let mut f32_pcm = body(text, seed);
    f32_pcm["response_format"] = json!("pcm_f32");
    let raw = bytes(speech(&server, f32_pcm).await).await;
    let floats = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        floats,
        expected.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
    );
    // `top_k: 0` is no cap: the same audio as without it.
    let mut uncapped = body(text, seed);
    uncapped["top_k"] = json!(0);
    let (_, again) = wav_samples(&bytes(speech(&server, uncapped).await).await);
    assert_eq!(again, wav);

    // A cloned voice: the reference and its transcript.
    let reference = reference_wav(2.0);
    let (clone_seed, cloned) =
        complete_seed(&server.state, text, 0, Some((&reference, "hello world."))).await;
    let mut clone = body(text, clone_seed);
    clone["reference_audio"] = json!(reference);
    clone["reference_text"] = json!("hello world.");
    let response = speech(&server, clone.clone()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let (_, clone_wav) = wav_samples(&bytes(response).await);
    assert_eq!(clone_wav, pcm_i16(&cloned));
    // Without its transcript the reference is refused.
    clone.as_object_mut().unwrap().remove("reference_text");
    let response = speech(&server, clone).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_slice(&bytes(response).await).unwrap();
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Provide both"));

    // A stream: server-sent PCM chunks of the same audio, one segment.
    let mut streamed = body(text, seed);
    streamed["stream"] = json!(true);
    streamed["response_format"] = json!("pcm_i16");
    let response = speech(&server, streamed).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["X-Izwi-Streaming-Mode"], "chunked");
    let events = sse_events(&bytes(response).await);
    let mut stream_pcm = Vec::new();
    for event in events.iter().filter(|e| e["event"] == "audio.chunk") {
        assert_eq!(event["sample_rate"], SAMPLE_RATE);
        let chunk = base64::engine::general_purpose::STANDARD
            .decode(event["audio_base64"].as_str().unwrap())
            .unwrap();
        stream_pcm.extend(
            chunk
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| i16::from_le_bytes(*pair)),
        );
    }
    assert!(
        events
            .iter()
            .filter(|e| e["event"] == "audio.chunk")
            .count()
            > 1
    );
    assert_eq!(stream_pcm, wav);
    let done = events.iter().find(|e| e["event"] == "audio.done").unwrap();
    assert_eq!(done["timing"]["token_unit"], "semantic_frames");
    assert_eq!(done["timing"]["planned_segment_count"], 1);
    assert_eq!(done["stop_reason"], "stop");

    // Long text: at least three chunks of the family's chunker, each
    // carrying its history, spoken one after another with no gap.
    let long = "one two three hello world. ".repeat(30);
    assert!(plan_chunks(&long, false).len() >= 3);
    let (long_seed, long_expected) = complete_seed(&server.state, &long, 100, None).await;
    let response = speech(&server, body(&long, long_seed)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let tokens: usize = response.headers()["X-Tokens-Generated"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    let (_, long_wav) = wav_samples(&bytes(response).await);
    assert_eq!(long_wav, pcm_i16(&long_expected));
    assert!(
        tokens >= long_wav.len() / STUB_SAMPLES_PER_FRAME,
        "{tokens}"
    );
    // Its chunks without their history speak otherwise: the history
    // reached every prompt after the first.
    let mut alone = Vec::new();
    for (index, chunk) in plan_chunks(&long, false).iter().enumerate() {
        match chunk_session_pcm(&server.state, chunk, long_seed + index as u64, None).await {
            Ok((pcm, _)) => alone.extend(pcm),
            Err(error) if error.to_string().contains("frame_limit") => {}
            Err(error) => panic!("{error}"),
        }
    }
    assert_ne!(pcm_i16(&alone), long_wav);
    // The stream reports its chunks as its planned segments.
    let mut long_stream = body(&long, long_seed);
    long_stream["stream"] = json!(true);
    let events = sse_events(&bytes(speech(&server, long_stream).await).await);
    let done = events.iter().find(|e| e["event"] == "audio.done").unwrap();
    assert_eq!(
        done["timing"]["planned_segment_count"],
        plan_chunks(&long, false).len()
    );

    // The frame budget: a take that does not end within it fails, it is
    // never cut short.
    let mut cut = None;
    for seed in 0..32 {
        let mut short = body(text, seed);
        short["max_tokens"] = json!(2);
        let response = speech(&server, short).await;
        if response.status() != StatusCode::OK {
            let message = String::from_utf8_lossy(&bytes(response).await).to_string();
            assert!(message.contains("frame_limit"), "{message}");
            cut = Some(seed);
            break;
        }
    }
    assert!(cut.is_some(), "a seed that runs past two frames");
    server
        .state
        .runtime
        .unload_model(VARIANT)
        .await
        .expect("unload");
}

/// A durable first-party record of long text: each chunk runs as its own
/// stage attempt (the stage yields at every chunk boundary), so the
/// history the next chunk needs comes back from the checkpoint. The
/// record's audio is the chunk session's.
///
/// The worker's stage future nests the whole durable speech path; an
/// unoptimized test build polls it deeper than tokio's default 2 MiB worker
/// stack, so this test's runtime gives its workers more.
#[test]
fn fish_drama3_durable_records_carry_history_across_stage_attempts() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(32 << 20)
        .enable_all()
        .build()
        .expect("a runtime");
    runtime
        .block_on(runtime.spawn(durable_records_carry_history_across_stage_attempts()))
        .expect("the test task");
}

async fn durable_records_carry_history_across_stage_attempts() {
    use crate::batch_runtime::worker::{BatchWorkerConfig, BatchWorkerRunner};
    let _load = LOADS.lock().await;
    let server = server("durable");
    server
        .state
        .runtime
        .load_model(VARIANT)
        .await
        .expect("load");
    let long = "one two three hello world. ".repeat(30);
    let chunks = plan_chunks(&long, false).len();
    assert!(chunks >= 3);
    let (seed, expected) = complete_seed(&server.state, &long, 200, None).await;

    let response = send(
        &server,
        Method::POST,
        "/v1/text-to-speech",
        Some(json!({
            "model_id": "FishAudio-Drama-3",
            "text": long,
            "temperature": TEMPERATURE,
            "max_tokens": BUDGET,
            "seed": seed,
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let record: Value = serde_json::from_slice(&bytes(response).await).unwrap();
    let id = record["id"].as_str().unwrap().to_string();

    let state = &server.state;
    let mut config = BatchWorkerConfig::local("drama3-durable-test");
    config.queue_names = crate::roles::local::local_batch_worker_queue_names();
    config.capabilities = vec!["tts".to_string()];
    config.stage_kinds = vec![crate::api::speech_history::BATCH_TTS_STAGE_KIND.to_string()];
    let backend = state.runtime.backend_context();
    config.resources = crate::roles::local::local_batch_worker_resources(
        backend.backend_kind,
        backend.device.capabilities.available_memory_bytes,
    );
    config.resources.concurrency_slots = 1;
    let worker = BatchWorkerRunner::new(
        state.batch_runtime_store.clone(),
        vec![crate::api::speech_history::batch_tts_stage_executor(
            state.clone(),
        )],
        config,
        state.batch_worker_health.clone(),
    )
    .with_runtime_observer(state.runtime.clone())
    .with_artifact_store(state.artifact_store.clone());
    let mut attempts = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    loop {
        let record: Value = serde_json::from_slice(
            &bytes(
                send(
                    &server,
                    Method::GET,
                    &format!("/v1/text-to-speech/{id}"),
                    None,
                )
                .await,
            )
            .await,
        )
        .unwrap();
        match record["processing_status"].as_str() {
            Some("ready") => break,
            Some("failed") => panic!("the record failed: {record}"),
            _ => {}
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the record never finished"
        );
        if worker.run_once().await.expect("a worker pass") {
            attempts += 1;
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
    // One stage attempt per chunk.
    assert_eq!(attempts, chunks);
    let response = send(
        &server,
        Method::GET,
        &format!("/v1/text-to-speech/{id}/audio"),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let (rate, wav) = wav_samples(&bytes(response).await);
    assert_eq!(rate, SAMPLE_RATE);
    assert_eq!(wav, pcm_i16(&expected));
    server
        .state
        .runtime
        .unload_model(VARIANT)
        .await
        .expect("unload");
}

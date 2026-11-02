//! Explicitly invoked, real-weight qualification of the managed streaming path.
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use base64::Engine as _;
use tokio::sync::mpsc;

use crate::backends::BackendPreference;
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::runtime::{GenerationRequest, RuntimeService};
use crate::{ContextLengthPreference, EngineConfig};

use super::FishS2DacConfig;

const TARGET: &str = "The scientist opened the window and listened to the rain. Every drop made a different sound on the leaves below. After a moment she smiled, closed her notebook, and decided to take a walk through the garden. By the time she returned, the clouds had cleared and the evening sky was full of stars.";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Fish weights, reference audio and explicit IZWI_FISH_S2_BACKEND"]
async fn fish_s2_real_runtime_streaming_emits_pcm_before_completion() -> Result<()> {
    let variant = ModelVariant::FishAudioS2Pro;
    #[cfg(feature = "metal")]
    let _lease = crate::backends::metal_test_device_or_skip(
        "fish_s2_real_runtime_streaming_emits_pcm_before_completion",
    );
    let model_dir = PathBuf::from(required("IZWI_FISH_S2_MODEL_DIR")?);
    if model_dir.file_name().and_then(|name| name.to_str()) != Some(variant.dir_name()) {
        return Err(Error::InvalidInput(format!(
            "IZWI_FISH_S2_MODEL_DIR must end in {}",
            variant.dir_name()
        )));
    }
    // Qualification must name the backend; an unavailable target cannot silently
    // become a successful CPU run and certify a CUDA/Metal release.
    let backend_name = required("IZWI_FISH_S2_BACKEND")?;
    let backend = match backend_name.as_str() {
        "cpu" => BackendPreference::Cpu,
        "cuda" => BackendPreference::Cuda,
        "metal" => BackendPreference::Metal,
        _ => {
            return Err(Error::InvalidInput(
                "Qualification requires cpu, cuda or metal; auto is not accepted".into(),
            ))
        }
    };
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let custom = env::var_os("IZWI_FISH_S2_REFERENCE_WAV");
    let reference = custom
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| fixtures.join("fox.wav"));
    let transcript = match env::var("IZWI_FISH_S2_REFERENCE_TEXT") {
        Ok(text) => text,
        Err(_) if custom.is_some() => {
            return Err(Error::InvalidInput(
                "Custom reference requires IZWI_FISH_S2_REFERENCE_TEXT".into(),
            ))
        }
        Err(_) => std::fs::read_to_string(fixtures.join("fox.md"))?,
    };
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: model_dir
            .parent()
            .ok_or_else(|| Error::InvalidInput("Missing model parent".into()))?
            .to_path_buf(),
        backend,
        max_sequence_length: ContextLengthPreference::explicit(4096)?,
        ..EngineConfig::default()
    })?;
    runtime.load_model(variant).await?;
    let mut request = GenerationRequest::new(TARGET).with_model_variant(variant);
    request.config.streaming = true;
    request.config.options.max_tokens = 512;
    request.config.options.temperature = 0.0;
    request.config.options.top_p = 1.0;
    request.reference_audio =
        Some(base64::engine::general_purpose::STANDARD.encode(std::fs::read(reference)?));
    request.reference_text = Some(transcript);
    let request_id = request.id.clone();
    let (tx, mut rx) = mpsc::channel(2);
    let completed = AtomicBool::new(false);
    let started = Instant::now();
    let producer = async {
        let result = runtime.generate_streaming(request, tx).await;
        completed.store(true, Ordering::Release);
        result
    };
    let consumer = async {
        let mut pcm = Vec::new();
        let mut arrivals_ms = Vec::new();
        let mut before_completion = 0;
        let mut terminal = None;
        let mut previous_sequence = None;
        let sample_rate = FishS2DacConfig::current().sample_rate;
        while let Some(chunk) = rx.recv().await {
            assert!(terminal.is_none(), "output after terminal marker");
            assert_eq!(chunk.request_id, request_id);
            assert_eq!(chunk.sample_rate, sample_rate);
            if let Some(previous) = previous_sequence {
                assert_eq!(chunk.sequence, previous + 1);
            }
            previous_sequence = Some(chunk.sequence);
            if !chunk.samples.is_empty() {
                assert!(chunk.samples.iter().all(|sample| sample.is_finite()));
                arrivals_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                if !chunk.is_final && !completed.load(Ordering::Acquire) {
                    before_completion += 1;
                }
                pcm.extend_from_slice(&chunk.samples);
            }
            if chunk.is_final {
                terminal = Some(
                    chunk
                        .stats
                        .expect("terminal cumulative statistics, even for empty PCM"),
                );
            }
        }
        assert!(
            before_completion >= 2,
            "expected at least two PCM chunks before runtime completion, got {before_completion}"
        );
        assert!(
            pcm.iter().any(|sample| sample.abs() > 1e-5),
            "silent waveform"
        );
        let stats = terminal.expect("exactly one terminal marker");
        assert_eq!(
            pcm.len(),
            stats.tokens_generated * FishS2DacConfig::current().samples_per_frame()?
        );
        let duration_secs = pcm.len() as f64 / f64::from(sample_rate);
        let execution_rtf = f64::from(stats.generation_time_ms) / 1000.0 / duration_secs;
        assert!(stats.generation_time_ms > 0.0);
        assert!((f64::from(stats.rtf) - execution_rtf).abs() < 1e-5);
        let mut gaps = arrivals_ms
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();
        gaps.sort_by(f64::total_cmp);
        let percentile = |p: f64| {
            gaps.get(((gaps.len().saturating_sub(1)) as f64 * p).ceil() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        eprintln!(
            "Fish streaming qualification: {}",
            serde_json::json!({
                "backend": backend_name, "request_id": request_id, "target_text": TARGET,
                "temperature": 0.0, "max_frames":512, "context":4096,
                "pcm_chunks": arrivals_ms.len(), "chunks_before_runtime_completion": before_completion,
                "semantic_frames":stats.tokens_generated, "samples":pcm.len(), "sample_rate":sample_rate,
                "audio_duration_secs":duration_secs, "server_ttfa_ms":arrivals_ms[0],
                "gap_p50_ms":percentile(0.5), "gap_p95_ms":percentile(0.95), "gap_p99_ms":percentile(0.99),
                "execution_rtf":execution_rtf,
                "request_to_last_pcm_rtf":arrivals_ms.last().unwrap()/1000.0/duration_secs,
                "timing_note":"Model load excluded; reference preparation included in request-to-PCM timings. Runtime completion ordering does not independently timestamp AR completion."
            })
        );
        if let Some(path) = env::var_os("IZWI_FISH_S2_SMOKE_OUTPUT_WAV") {
            let mut writer = hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .map_err(|error| Error::AudioError(error.to_string()))?;
            for sample in pcm {
                writer
                    .write_sample(sample)
                    .map_err(|error| Error::AudioError(error.to_string()))?;
            }
            writer
                .finalize()
                .map_err(|error| Error::AudioError(error.to_string()))?;
        }
        Ok::<(), Error>(())
    };
    let (generated, consumed) = tokio::join!(producer, consumer);
    let unloaded = runtime.unload_model(variant).await;
    generated?;
    consumed?;
    unloaded
}

fn required(name: &str) -> Result<String> {
    env::var(name)
        .map_err(|_| Error::InvalidInput(format!("Set {name} for Fish streaming qualification")))
}

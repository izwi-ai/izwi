//! Runtime-level streaming-synthesis smoke for the Kokoro family (ignored,
//! env-gated, same pattern as the Fish S2 streaming smoke): documents what
//! `RuntimeService::generate_streaming` emits for a direct-streaming TTS
//! family, which differs from the engine path's chunk cadence.
//!
//! Measured shape (2026-09-26, CPU, one-sentence fixture): the whole
//! utterance arrives as ONE non-empty chunk carrying `is_final: true` — the
//! engine path's separate empty terminal-stats chunk does not exist here, so
//! session consumers must treat payload-bearing final frames as normal.
//! CPU latency is roughly RTF 18 (a 2.8 s utterance takes ~52 s).

use izwi_core::{GenerationRequest, RuntimeService, WorkloadClass};
use std::time::Duration;

#[tokio::test]
#[ignore = "requires the local Kokoro artifact (IZWI_REAL_TTS_MODELS_DIR)"]
async fn kokoro_generate_streaming_emits_pcm_and_a_final_chunk() {
    let models_root = std::env::var("IZWI_REAL_TTS_MODELS_DIR")
        .expect("set IZWI_REAL_TTS_MODELS_DIR to the local models root");
    let engine = izwi_core::EngineConfig {
        models_dir: std::path::PathBuf::from(&models_root),
        ..Default::default()
    };
    let runtime =
        RuntimeService::new_assigned(engine, izwi_core::backends::RuntimeDeviceAssignment::Cpu)
            .expect("runtime constructs");
    runtime
        .load_model(izwi_core::ModelVariant::Kokoro82M)
        .await
        .expect("kokoro loads");
    assert_eq!(runtime.sample_rate().await, 24_000);

    let request = GenerationRequest::new("The quick brown fox jumps over the lazy dog.")
        .with_model_variant(izwi_core::ModelVariant::Kokoro82M)
        .with_runtime_context(
            izwi_core::RuntimeRequestContext::new(WorkloadClass::Realtime)
                .with_deadline(std::time::Instant::now() + Duration::from_secs(300)),
        );
    let (tx, mut rx) = tokio::sync::mpsc::channel::<izwi_core::AudioChunk>(32);
    let generation = tokio::spawn(async move { runtime.generate_streaming(request, tx).await });

    let mut samples = 0usize;
    let mut saw_final = false;
    while let Some(chunk) = rx.recv().await {
        samples += chunk.samples.len();
        saw_final = saw_final || chunk.is_final;
    }
    generation.await.unwrap().expect("generation succeeds");
    assert!(saw_final, "the stream carries a final-flagged chunk");
    assert!(samples > 0, "the stream carries synthesized PCM");
}

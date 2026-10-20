//! Real-checkpoint probe for Parakeet's long-form atomic graph (ignored,
//! env-gated). The diarization pipeline loads its ASR with the long-form flag
//! and runs it through the atomic invocation stage; ordinary transcription
//! uses the retained sequence graph. Both must produce the same text.

use izwi_core::{DiarizationConfig, DiarizationPipelineOptions, RuntimeService};

const PARAKEET: &str = "Parakeet-TDT-0.6B-v3";

fn models_root() -> std::path::PathBuf {
    std::env::var_os("IZWI_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("HOME is set");
            std::path::PathBuf::from(home).join("Library/Application Support/izwi/models")
        })
}

#[tokio::test]
#[ignore = "requires local Parakeet, Sortformer and forced-aligner checkpoints (IZWI_MODELS_DIR)"]
async fn parakeet_long_form_pipeline_matches_ordinary_transcription() {
    let engine = izwi_core::EngineConfig {
        models_dir: models_root(),
        ..Default::default()
    };
    let runtime =
        RuntimeService::new_assigned(engine, izwi_core::backends::RuntimeDeviceAssignment::Cpu)
            .expect("runtime constructs");
    let audio = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/fox.wav"))
        .expect("fox.wav fixture");

    let ordinary = runtime
        .asr_transcribe_bytes(&audio, Some(PARAKEET), None)
        .await
        .expect("ordinary Parakeet transcription");
    let pipeline = runtime
        .diarize_with_transcript_bytes(
            &audio,
            None,
            Some(PARAKEET),
            None,
            &DiarizationConfig::default(),
            &DiarizationPipelineOptions::default(),
        )
        .await
        .expect("diarization pipeline with the default Parakeet ASR");

    eprintln!("ordinary: {:?}", ordinary.text);
    eprintln!("pipeline: {:?}", pipeline.asr_text);
    assert!(!ordinary.text.trim().is_empty());
    assert_eq!(pipeline.asr_text.trim(), ordinary.text.trim());
}

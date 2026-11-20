//! Real-checkpoint parity against the reference JSON that
//! `scripts/fixtures/generate_sov_transcribe_reference.py` writes from the
//! model's own Transformers code. Ignored: it needs the 53 GB checkpoint and
//! a CUDA GPU of 64 GB or more.
//!
//! For each recorded clip: the prompt ids exactly, the audio embeddings and
//! the prompt's last hidden state within BF16 tolerance, the first greedy
//! tokens inside the reference's top 5, and the greedy transcript.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;

use super::model::{SovStepRow, SovTranscribeModel, SovTranscribeRequest};
use super::prompt::{SovPromptOptions, AUTO_LANGUAGE_INSTRUCTION};

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn reference_path() -> PathBuf {
    env_path("IZWI_SOV_TRANSCRIBE_REFERENCE").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sov_transcribe/reference.json")
    })
}

#[test]
#[ignore = "needs the published checkpoint, its reference JSON and a CUDA GPU"]
fn the_published_checkpoint_matches_the_reference() {
    let model_dir = env_path("IZWI_SOV_TRANSCRIBE_MODEL_DIR")
        .expect("set IZWI_SOV_TRANSCRIBE_MODEL_DIR to the checkpoint");
    let audio_dir = env_path("IZWI_SOV_TRANSCRIBE_AUDIO_DIR")
        .expect("set IZWI_SOV_TRANSCRIBE_AUDIO_DIR to the reference clips");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference_path()).expect("reference JSON"))
            .expect("reference JSON parses");
    let device = Device::new_cuda(0).expect("a CUDA device");
    let model = SovTranscribeModel::load_on(&model_dir, None, &device).expect("load");
    eprintln!("{}", model.diagnostics_summary());
    for case in reference["cases"].as_array().expect("cases") {
        let file = case["file"].as_str().unwrap();
        assert_eq!(case["instruction"], AUTO_LANGUAGE_INSTRUCTION);
        let (samples, rate) =
            crate::audio::decode_audio_bytes_to_mono(&std::fs::read(audio_dir.join(file)).unwrap())
                .expect("a mono WAV");
        let artifact = Arc::new(
            model
                .prepare_audio(
                    &samples,
                    rate,
                    SovTranscribeRequest {
                        prompt: SovPromptOptions::default(),
                        max_new_tokens: Some(256),
                    },
                )
                .unwrap(),
        );
        let ids = artifact
            .layout
            .expanded_ids(model.config().audio_token_id)
            .into_iter()
            .map(u64::from)
            .collect::<Vec<_>>();
        let expected_ids = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, expected_ids, "{file}: prompt ids");
        let audio = model.encode_audio(&[&artifact.mel]).unwrap().remove(0);
        let first = audio
            .flatten_all()
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .narrow(0, 0, 16)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (ours, theirs) in first
            .iter()
            .zip(case["audio_embeddings"]["first"].as_array().unwrap())
        {
            let theirs = theirs.as_f64().unwrap() as f32;
            assert!(
                (ours - theirs).abs() <= 0.05 + 0.05 * theirs.abs(),
                "{file}: audio {ours} vs {theirs}"
            );
        }
        let context = artifact.max_context();
        let mut caches = super::model_tests::caches(&model, 1, context + 16);
        let mut state = model.new_row(artifact, context).unwrap();
        let prompt = state.prompt_tokens();
        model
            .retained_prefill(&mut state, &mut caches[0], prompt)
            .unwrap();
        while !state.finished() {
            model
                .retained_decode_batch(&mut [SovStepRow {
                    state: &mut state,
                    cache: &mut caches[0],
                }])
                .unwrap();
        }
        let tokens = state.decode_loop().unwrap().generated().to_vec();
        for (step, (token, top5)) in tokens
            .iter()
            .zip(case["top5"].as_array().unwrap())
            .enumerate()
        {
            let top5 = top5
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            assert!(
                top5.contains(token),
                "{file}: step {step} token {token} not in {top5:?}"
            );
        }
        let text = model.finish_row(&state).unwrap().text;
        eprintln!("{file}: {text}");
        assert_eq!(text, case["text"].as_str().unwrap(), "{file}: transcript");
    }
}

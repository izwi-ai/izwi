//! Nemotron 3.5 Transcribe 30B-A4B FP8: NVIDIA's speech-augmented language
//! model, as its own family.
//!
//! The model is a NeMo SpeechLM bundle:
//! * a log-mel front end (`frontend`);
//! * a Parallel Expert Encoder that runs a 32-layer Transformer ASR encoder
//!   and the Nemotron-3-Diarization Sortformer as a streaming speaker
//!   encoder over 40 s windows, and fuses them with a speaker kernel
//!   (`speech_transformer`, `sortformer`, `perception`);
//! * a Nemotron-H 30B-A3B decoder: Mamba2, sigmoid-routed relu² MoE and GQA
//!   attention without RoPE (`decoder`);
//! * a CTC timestamp head (`ctc`).
//!
//! It is separate from `nemotron` (the FastConformer RNNT streaming ASR) and
//! from `sortformer` (the standalone diarizer). Nothing here imports another
//! family: the family may use `kernels`, `models::shared`, `kv`, `backends`,
//! `engine` and `audio`, and builds its own state contract. The isolation
//! test at the bottom of this file holds that line.

pub mod checkpoint;
pub mod config;
pub mod decoder;
pub mod frontend;
pub mod linear;
pub mod model;
pub mod perception;
pub mod precision;
pub mod prompt;
pub mod sortformer;
pub mod speech_transformer;
pub mod transcript;

#[cfg(test)]
mod model_tests;
#[cfg(test)]
pub(crate) mod synthetic;

pub use model::{
    planned_decode_tokens, NemotronPreparedArtifact, NemotronRetainedCheckpoint,
    NemotronRetainedState, NemotronStepRow, NemotronTranscribeModel, NemotronTranscribeRequest,
    NemotronTranscription, MAX_AUDIO_SECONDS, MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS,
};
pub use prompt::{NemotronPromptOptions, LANGUAGES as NEMOTRON_TRANSCRIBE_LANGUAGES};
pub use transcript::NemotronFinishReason;

#[cfg(test)]
mod isolation_tests {
    use std::path::{Path, PathBuf};

    /// Walks the family's sources: the only architecture path any of
    /// them may name is this family's own.
    #[test]
    fn the_family_imports_no_other_architecture() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/models/architectures/nemotron_transcribe");
        fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        let mut files = Vec::new();
        sources(&root, &mut files);
        assert!(files.len() >= 2);
        let needle = concat!("architectures", "::");
        let own = concat!("architectures", "::", "nemotron_transcribe");
        for file in files {
            let source = std::fs::read_to_string(&file).unwrap();
            for (offset, _) in source.match_indices(needle) {
                assert!(
                    source[offset..].starts_with(own),
                    "{} names another architecture at byte {offset}",
                    file.display()
                );
            }
            // `super::super::` would climb into `architectures` itself.
            assert!(
                !source.contains(concat!("super::super::super::", "architectures")),
                "{}",
                file.display()
            );
        }
    }
}

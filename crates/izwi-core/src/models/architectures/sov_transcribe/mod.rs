//! SOV-Transcribe-1: Salesforce's speech-augmented language model for
//! non-streaming transcription, as its own family.
//!
//! The model is three parts:
//! * a Whisper log-mel front end over the whole clip, cut into 30 s windows
//!   (`frontend`);
//! * the Whisper large-v3 encoder, one batch row per window, and a
//!   projector that stacks four frames into one audio token (`encoder`);
//! * the Gemma 4 26B-A4B decoder: sliding and full attention layers with
//!   different head geometries, and in every layer a dense MLP beside 128
//!   routed experts (`decoder`).
//!
//! It is served on CUDA only (`precision`); the CPU runs a tiny synthetic
//! checkpoint as the test reference. It is separate from `whisper` (whose
//! encoder shape it shares) and from `gemma3`. Nothing here imports another
//! family: the family may use `kernels`, `models::shared`, `kv`,
//! `backends`, `engine` and `audio`, and builds its own state contract. The
//! isolation test at the bottom of this file holds that line.

pub mod checkpoint;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod frontend;
pub mod model;
pub mod precision;
pub mod prompt;
pub mod residency;
pub mod transcript;

#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
pub(crate) mod synthetic;

pub use model::{
    decode_quantum, planned_decode_tokens, prefill_span, SovPrefillRow, SovPreparedArtifact,
    SovRetainedCheckpoint, SovRetainedState, SovStepRow, SovTranscribeModel, SovTranscribeRequest,
    SovTranscription, BATCHED_PREFILL_ENV, DECODE_QUANTUM_ENV, ENCODER_BATCH_ENV,
    MAX_AUDIO_SECONDS, MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS, PREFILL_SPAN_ENV,
};
pub use precision::{cuda_only_message, require_serving_backend};
pub use prompt::{SovPromptOptions, LANGUAGES as SOV_TRANSCRIBE_LANGUAGES};
pub use transcript::SovFinishReason;

#[cfg(test)]
mod isolation_tests {
    use std::path::Path;

    fn visit(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }

    /// The family imports no other architecture: every `architectures::`
    /// path and every `super::super::<family>` reach stays inside
    /// `sov_transcribe`.
    #[test]
    fn the_family_imports_no_other_architecture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures");
        let families = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_dir())
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .filter(|name| name != "sov_transcribe")
            .collect::<Vec<_>>();
        assert!(families.contains(&"whisper".to_string()));
        let mut files = Vec::new();
        visit(&root.join("sov_transcribe"), &mut files);
        assert!(files.len() >= 15);
        for file in files {
            let source = std::fs::read_to_string(&file).unwrap();
            for (number, line) in source.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                for family in &families {
                    for pattern in [
                        format!("architectures::{family}::"),
                        format!("architectures::{family};"),
                        format!("architectures::{{{family}"),
                        format!("super::super::{family}::"),
                        format!("super::super::super::{family}::"),
                    ] {
                        assert!(
                            !code.contains(&pattern),
                            "{}:{} imports the {family} family: {line}",
                            file.display(),
                            number + 1
                        );
                    }
                }
            }
        }
    }
}

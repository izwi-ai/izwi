//! Native Reson8 Rumble ASR (Rumble-2-Flash).
//!
//! Rumble is a family of its own: a windowed audio tower and a two-layer
//! projector feed a Qwen3 text trunk, over a pruned vocabulary and a prompt
//! contract with transcript style control tokens. The checkpoint is
//! published in the Transformers-native layout.
//!
//! Nothing here calls the Qwen3-ASR family's module, and that family does
//! not call this one; a test below keeps it so. The decoder is the
//! shared Qwen3 text trunk in `qwen3::core`, which chat, TTS and VibeVoice
//! also load; the audio tower, projector, front end, tokenizer, prompt and
//! transcript handling are Rumble's own.
//!
//! Every request runs as atomic invocations on a lifecycle-owned paged KV
//! lease: audio of up to 30 s is one invocation, and longer audio is cut
//! into chunks by the generic ASR route.

mod audio_tower;
mod checkpoint;
mod config;
mod frontend;
mod model;
mod precision;
mod prompt;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod synthetic;
#[cfg(test)]
mod tests;
mod tokenizer;
mod transcript;

pub(crate) use audio_tower::RUMBLE_AUDIO_POSITIONS;
pub use checkpoint::ensure_rumble_artifacts;
pub use config::{RumbleAudioConfig, RumbleConfig, RumblePreprocessorConfig, RumbleTextConfig};
pub use frontend::{
    RumbleAudioFeatures, RUMBLE_AUDIO_TOKENS_PER_SECOND, RUMBLE_MAX_AUDIO_SECONDS,
    RUMBLE_SAMPLE_RATE,
};
pub(crate) use model::RumblePhysicalStateSpec;
pub use model::{
    RumbleAsrModel, RumbleGenerationOptions, RumbleRequestOptions, RumbleTranscription,
    RUMBLE_DEFAULT_MAX_NEW_TOKENS, RUMBLE_SEQUENCE_TOKENS,
};
pub(crate) use precision::rumble_weight_dtypes;
pub use prompt::{
    resolve_rumble_language, RumblePrompt, RumblePromptOptions, RumbleTranscriptStyle,
    RUMBLE_DETECTED_LANGUAGES, RUMBLE_FORCED_LANGUAGES, RUMBLE_MAX_CONTEXT_TOKENS,
};
pub use tokenizer::{RumbleSpecialTokens, RumbleTokenizer};
pub use transcript::{split_rumble_output, RumbleOutput};

/// Reject a request Rumble cannot serve before it is admitted: a context
/// prompt longer than the system turn holds. The check is by characters
/// (the tokenizer is not loaded at validation time); the exact token limit
/// is enforced when the prompt is built.
pub fn validate_rumble_request(options: &RumbleRequestOptions) -> crate::error::Result<()> {
    const MAX_CONTEXT_CHARS: usize = RUMBLE_MAX_CONTEXT_TOKENS * 8;
    if let Some(prompt) = options.prompt.as_deref() {
        if prompt.trim().chars().count() > MAX_CONTEXT_CHARS {
            return Err(crate::error::Error::InvalidInput(format!(
                "Rumble context prompt is too long: at most {MAX_CONTEXT_CHARS} characters"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod family_separation {
    use std::fs;
    use std::path::Path;

    fn rust_sources(dir: &Path, sources: &mut Vec<(String, String)>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_sources(&path, sources);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push((
                    path.display().to_string(),
                    fs::read_to_string(&path).unwrap(),
                ));
            }
        }
    }

    /// Rumble and Qwen3-ASR are separate families: neither module names the
    /// other's path.
    #[test]
    fn rumble_and_qwen3_asr_do_not_import_each_other() {
        let architectures = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures");
        let mut rumble = Vec::new();
        rust_sources(&architectures.join("rumble"), &mut rumble);
        assert!(rumble.len() >= 8, "rumble sources not found");
        let forbidden = ["qwen3::", "asr"].concat();
        for (path, source) in &rumble {
            assert!(
                !source.contains(&forbidden),
                "{path} reaches into the Qwen3-ASR family"
            );
        }
        let mut qwen3_asr = Vec::new();
        rust_sources(&architectures.join("qwen3/asr"), &mut qwen3_asr);
        assert!(!qwen3_asr.is_empty(), "qwen3 asr sources not found");
        let forbidden = ["architectures::", "rumble"].concat();
        for (path, source) in &qwen3_asr {
            assert!(
                !source.contains(&forbidden) && !source.contains("Rumble"),
                "{path} reaches into the Rumble family"
            );
        }
    }
}

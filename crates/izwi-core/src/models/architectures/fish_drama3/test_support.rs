//! The synthetic checkpoint (§10.1) as other crates' tests load it: a model
//! directory with its tokenizer, marked for the lifecycle, which a runtime
//! then loads with the stub codec as it would the published checkpoint.
//!
//! Built for this crate's tests and, through the `test-support` feature,
//! for other crates' tests (the server's speech routes run on it). A
//! release build has neither, so it never loads anything but the pinned
//! geometry.

use std::path::Path;

use serde_json::{json, Value};

use super::codec::FishDrama3DacStreamState;
use super::model::{FishDrama3CancelCheck, FishDrama3FrameCodec};
use super::prompt::FishDrama3Codes;
use super::synthetic::{write_geometry_at, SyntheticGeometry, LIFECYCLE_MARKER};
use super::tokenizer::{semantic_token, speaker_token};
use crate::error::Result;

/// Samples the stub codec makes per frame, as the published codec does.
pub const STUB_SAMPLES_PER_FRAME: usize = 2048;

/// A stateless codec: each frame's 2,048 samples come from its codes alone.
pub struct StubCodec;

/// One frame's samples from its codes.
pub(crate) fn stub_frame(codes: &[u32]) -> impl Iterator<Item = f32> + '_ {
    let seed = codes
        .iter()
        .enumerate()
        .map(|(k, &code)| (k as u32 + 1) * code)
        .sum::<u32>() as f32;
    (0..STUB_SAMPLES_PER_FRAME).map(move |j| (seed * 0.01 + j as f32 * 1e-3).sin() * 0.5)
}

impl FishDrama3FrameCodec for StubCodec {
    fn sample_rate(&self) -> u32 {
        44_100
    }

    fn samples_per_frame(&self) -> Result<usize> {
        Ok(STUB_SAMPLES_PER_FRAME)
    }

    fn codebooks(&self) -> usize {
        10
    }

    fn stream_history_bound_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    fn stream_frames(
        &self,
        _state: &mut FishDrama3DacStreamState,
        codebooks: &[Vec<u32>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        check()?;
        let frames = codebooks[0].len();
        Ok((0..frames)
            .flat_map(|f| {
                let codes = codebooks.iter().map(|row| row[f]).collect::<Vec<_>>();
                stub_frame(&codes).collect::<Vec<_>>()
            })
            .collect())
    }

    fn stream_frames_batch(
        &self,
        states: &mut [FishDrama3DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        states
            .iter_mut()
            .zip(codebooks)
            .map(|(state, codes)| self.stream_frames(state, codes, check))
            .collect()
    }

    fn encode_reference(
        &self,
        samples: &[f32],
        _sample_rate: u32,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3Codes> {
        check()?;
        let frames = self.reference_frames(samples.len(), 0)?;
        Ok(FishDrama3Codes::new(
            (0..10)
                .map(|k| {
                    (0..frames)
                        .map(|f| ((k * 7 + f * 13) % 64) as u32)
                        .collect()
                })
                .collect(),
        ))
    }

    /// A frame per 2,048 samples, the sample rate aside.
    fn reference_frames(&self, samples: usize, _sample_rate: u32) -> Result<usize> {
        Ok(samples.div_ceil(STUB_SAMPLES_PER_FRAME))
    }
}

/// A WordLevel tokenizer laid out like the synthetic checkpoint's ids: pad
/// 468, `<|im_start|>` 469, eos 470, audio pad 480, `<|voice|>` 494,
/// speakers 496..=511, the semantic band 512..=575.
pub(crate) fn tokenizer_json() -> Value {
    let mut controls: Vec<(String, u32)> = vec![
        ("<|endoftext|>".into(), 468),
        ("<|im_start|>".into(), 469),
        ("<|im_end|>".into(), 470),
        ("<|audio_pad|>".into(), 480),
        ("<|voice|>".into(), 494),
        ("<|interleave|>".into(), 495),
    ];
    controls.extend((0..16).map(|index| (speaker_token(index), 496 + index as u32)));
    controls.extend((0..64).map(|code| (semantic_token(code), 512 + code as u32)));
    let words = [
        "system",
        "user",
        "assistant",
        "convert",
        "the",
        "provided",
        "text",
        "to",
        "speech",
        "reference",
        "following:",
        "hello",
        "world.",
        "Text:",
        "Speech:",
        "one",
        "two",
        "three",
    ];
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".into(), 0.into());
    for (index, word) in words.iter().enumerate() {
        vocab.insert((*word).into(), (index + 1).into());
    }
    for (token, id) in &controls {
        vocab.insert(token.clone(), (*id).into());
    }
    let added = controls
        .iter()
        .map(|(token, id)| {
            json!({
                "id": id, "content": token, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect::<Vec<_>>();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
}

/// The synthetic checkpoint with its tokenizer at `dir` (a runtime's
/// `models_dir/<variant dir>`), drawn from `seed`, marked for the
/// lifecycle: a runtime loads it as it would the published checkpoint,
/// under the synthetic config policy and with [`StubCodec`].
pub fn write_lifecycle_synthetic_model(dir: &Path, seed: u64) {
    write_lifecycle_synthetic_model_in(dir, seed, SyntheticGeometry::L0);
}

/// [`write_lifecycle_synthetic_model`] in `geometry`.
pub(crate) fn write_lifecycle_synthetic_model_in(
    dir: &Path,
    seed: u64,
    geometry: SyntheticGeometry,
) {
    write_geometry_at(dir, seed, geometry);
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
    std::fs::write(dir.join(LIFECYCLE_MARKER), b"").unwrap();
}

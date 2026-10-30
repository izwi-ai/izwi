//! Fish Audio Drama 3 (`fishaudio/drama-3`): a 35.9B-parameter
//! text-to-speech model with about 3B active per frame, as its own family.
//!
//! One frame is:
//! * a step of the trunk, Qwen3.5-35B-A3B's language model (30 Gated
//!   DeltaNet and 10 gated full-attention layers, every one a 256-expert
//!   top-8 MoE with FP8 block-scaled experts), whose main token is the
//!   frame's semantic code or `<|im_end|>`;
//! * the codebook-MTP tower (4 shared and 9 per-codebook dense layers at the
//!   trunk's width, each keeping its own KV over the whole sequence), which
//!   samples the 9 residual codes;
//! * the Fish S2 DAC codec, unchanged, which turns frames into 44.1 kHz
//!   audio.
//!
//! Prompts are ChatML with a reference-bracket system block: up to 8
//! reference clips with speaker tags, history turns with their audio, and
//! free-text performance tags.
//!
//! It is separate from Fish S2 (a dense dual-AR model with another tower,
//! prompt and checkpoint) and from the Qwen3.6 MoE chat family (another
//! checkpoint contract, M-RoPE, MTP and chat): sharing code with either
//! would let a change for one silently alter the other. Nothing here imports
//! another family: the family may use `kernels`, `models::shared`, `kv`,
//! `backends`, `engine` and `audio`, and keeps its own reader, trunk, tower,
//! codec copy and state contract. The isolation test at the bottom of this
//! file holds that line.

pub mod artifacts;
pub mod checkpoint;
pub mod chunking;
pub mod codec;
pub mod config;
pub mod embed;
pub mod head;
pub mod linear;
pub mod precision;
pub mod prompt;
pub mod reference_cache;
pub mod residency;
pub mod sampling;
#[cfg(test)]
pub(crate) mod synthetic;
pub mod tokenizer;
pub mod tower;
pub mod trunk;

pub use artifacts::{FISH_DRAMA3_REPO, FISH_DRAMA3_REVISION};
pub use checkpoint::FishDrama3Checkpoint;
pub use chunking::FishDrama3History;
pub use codec::{
    FishDrama3CodecArtifact, FishDrama3CodecSupport, FishDrama3DacConfig, FishDrama3DacDecoder,
};
pub use config::{FishDrama3Config, FishDrama3ConfigPolicy};
pub use embed::FishDrama3Embedding;
pub use head::FishDrama3CompactHead;
pub use linear::FishDrama3KernelPhase;
pub use precision::FishDrama3PrecisionReport;
pub use prompt::{FishDrama3Codes, FishDrama3HistoryTurn, FishDrama3Prompt, FishDrama3Reference};
pub use reference_cache::FishDrama3ReferenceAudio;
pub use residency::{FishDrama3CpuPrecision, FishDrama3Residency, FishDrama3WeightForm};
pub use sampling::{
    FishDrama3DrawPolicy, FishDrama3MainDraw, FishDrama3RowSampler, FishDrama3SamplerCheckpoint,
    FishDrama3SamplingParams,
};
pub use tokenizer::FishDrama3Tokenizer;
pub use tower::{
    FishDrama3Tower, FishDrama3TowerCheckpoint, FishDrama3TowerForcedRow, FishDrama3TowerOptions,
    FishDrama3TowerStepRow,
};
pub use trunk::{
    FishDrama3StepRow, FishDrama3Trunk, FishDrama3TrunkCheckpoint, FishDrama3TrunkOptions,
    FishDrama3TrunkState,
};

#[cfg(test)]
mod isolation_tests {
    use std::path::{Path, PathBuf};

    /// Walks the family's sources: the only architecture path any of
    /// them may name is this family's own.
    #[test]
    fn the_family_imports_no_other_architecture() {
        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures/fish_drama3");
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
        assert!(files.len() >= 4);
        // Every sibling family, so a relative `super::super::<family>::`
        // path is caught as well.
        let others = std::fs::read_dir(root.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter_map(|path| Some(path.file_stem()?.to_str()?.to_string()))
            .filter(|name| name != "mod" && name != "fish_drama3")
            .collect::<Vec<_>>();
        assert!(others.iter().any(|name| name == "fish_s2"));
        let needle = concat!("architectures", "::");
        let own = concat!("architectures", "::", "fish_drama3");
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
            for other in &others {
                let path = format!("{other}::");
                for (offset, _) in source.match_indices(&path) {
                    let before = source[..offset].chars().next_back();
                    assert!(
                        before.is_some_and(|c| c.is_alphanumeric() || c == '_'),
                        "{} names the {other} family at byte {offset}",
                        file.display()
                    );
                }
            }
        }
    }
}

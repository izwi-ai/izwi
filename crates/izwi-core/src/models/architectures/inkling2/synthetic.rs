//! A tiny Inkling 2 checkpoint on disk for runtime tests: the reference
//! fixture's weights (`tests/fixtures/inkling2_golden`, in the published
//! layout with both expert formats) with a byte-level tokenizer.
//!
//! Built for this crate's tests and, through the `test-support` feature,
//! for another crate's (the serving worker's process test). A release build
//! has none of it.

use std::path::Path;

use super::prompt::{END_SAMPLING, IMAGE_SLOT};
use super::reply::test_tokenizer::{test_tokenizer_json, test_tokenizer_sized};
use crate::artifacts::downloader::INKLING2_SYNTHETIC_MARKER;

/// ASCII byte tokens; the special tokens follow them.
pub(crate) const BYTE_TOKENS: u32 = 128;

/// Writes the checkpoint into `dir` (a model directory).
pub fn write_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // This crate's own fixture directory, also when another crate's test
    // calls in.
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_golden");
    std::fs::copy(
        golden.join("model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    // The end-of-sampling token is wherever the test tokenizer puts it.
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden.join("config.json")).unwrap()).unwrap();
    let end = test_tokenizer_sized(BYTE_TOKENS)
        .token_to_id(END_SAMPLING)
        .unwrap();
    config["eos_token_id"] = serde_json::json!(end);
    config["image_token_id"] = serde_json::json!(test_tokenizer_sized(BYTE_TOKENS)
        .token_to_id(IMAGE_SLOT)
        .unwrap());
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), test_tokenizer_json(BYTE_TOKENS)).unwrap();
    std::fs::write(dir.join(INKLING2_SYNTHETIC_MARKER), b"").unwrap();
}

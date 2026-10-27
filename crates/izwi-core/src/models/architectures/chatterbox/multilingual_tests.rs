//! The multilingual checkpoint against the English one.
//!
//! Upstream's multilingual loader reads `s3gen.pt` and `ve.pt` (pickled
//! state dicts) where the English one reads `s3gen.safetensors` and
//! `ve.safetensors`. izwi serves both editions from the safetensors, which
//! holds only if every tensor the loaders read is bit-identical; this census
//! checks that, and that the multilingual T3 matches the English geometry
//! but for its 2,454-token text tables.
//!
//! Run: `IZWI_CHATTERBOX_MODEL_DIR=… IZWI_CHATTERBOX_MULTILINGUAL_DIR=… cargo
//! test -p izwi-core --release --lib chatterbox::multilingual_tests --
//! --ignored --nocapture`

use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use super::config::{
    T3Config, MULTILINGUAL_T3_WEIGHTS_FILE, S3GEN_WEIGHTS_FILE, T3_WEIGHTS_FILE,
    VOICE_ENCODER_WEIGHTS_FILE,
};

fn dir(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable}"))
}

fn safetensors(path: PathBuf) -> BTreeMap<String, Tensor> {
    candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .into_iter()
        .collect()
}

fn pickled(path: PathBuf) -> BTreeMap<String, Tensor> {
    candle_core::pickle::read_all(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .into_iter()
        .collect()
}

fn same(a: &Tensor, b: &Tensor) -> bool {
    a.dims() == b.dims()
        && a.dtype() == b.dtype()
        && a.flatten_all()
            .and_then(|a| a.to_dtype(DType::F64))
            .and_then(|a| a.to_vec1::<f64>())
            .ok()
            == b.flatten_all()
                .and_then(|b| b.to_dtype(DType::F64))
                .and_then(|b| b.to_vec1::<f64>())
                .ok()
}

/// Every tensor of the English safetensors equals its pickled twin; keys
/// only one side has are reported (the pickles also carry buffers the
/// loaders rebuild, such as the tokenizer's mel filters).
#[test]
#[ignore = "needs both Chatterbox checkpoints"]
fn multilingual_s3gen_and_voice_encoder_equal_the_english_safetensors() {
    let english = dir("IZWI_CHATTERBOX_MODEL_DIR");
    let multilingual = dir("IZWI_CHATTERBOX_MULTILINGUAL_DIR");
    for (tensors, pickle) in [
        (S3GEN_WEIGHTS_FILE, "s3gen.pt"),
        (VOICE_ENCODER_WEIGHTS_FILE, "ve.pt"),
    ] {
        let a = safetensors(english.join(tensors));
        let b = pickled(multilingual.join(pickle));
        let only_safetensors = a.keys().filter(|k| !b.contains_key(*k)).collect::<Vec<_>>();
        let only_pickle = b.keys().filter(|k| !a.contains_key(*k)).collect::<Vec<_>>();
        let differing = a
            .iter()
            .filter(|(name, tensor)| b.get(*name).is_some_and(|other| !same(tensor, other)))
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        eprintln!(
            "{tensors} vs {pickle}: {} shared, {} differ, only safetensors {:?}, only pickle {:?}",
            a.len() - only_safetensors.len(),
            differing.len(),
            only_safetensors,
            only_pickle
        );
        assert!(
            only_safetensors.is_empty(),
            "{tensors} has tensors {pickle} lacks"
        );
        assert!(differing.is_empty(), "{tensors} differs: {differing:?}");
    }
}

/// The multilingual T3 has the English T3's tensors, with the same shapes
/// except the text tables, which have 2,454 rows.
#[test]
#[ignore = "needs both Chatterbox checkpoints"]
fn multilingual_t3_differs_only_in_its_text_tables() {
    let english = safetensors(dir("IZWI_CHATTERBOX_MODEL_DIR").join(T3_WEIGHTS_FILE));
    let multilingual =
        safetensors(dir("IZWI_CHATTERBOX_MULTILINGUAL_DIR").join(MULTILINGUAL_T3_WEIGHTS_FILE));
    assert_eq!(
        english.keys().collect::<Vec<_>>(),
        multilingual.keys().collect::<Vec<_>>()
    );
    let vocab = T3Config::multilingual().text_vocab_size;
    for (name, tensor) in &english {
        let other = &multilingual[name];
        if name.starts_with("text_emb.") || name.starts_with("text_head.") {
            assert_eq!(other.dims()[0], vocab, "{name}");
            assert_eq!(other.dims()[1..], tensor.dims()[1..], "{name}");
        } else {
            assert_eq!(other.dims(), tensor.dims(), "{name}");
        }
    }
}

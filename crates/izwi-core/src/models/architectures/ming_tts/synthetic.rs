//! A tiny Ming model with random weights, for tests that exercise the
//! whole pipeline without the checkpoint: the real layout at width 32.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{AddedToken, Tokenizer};

use super::config::{MingTtsConfig, PINNED_TOKENS};
use super::flow::tests::{tiny_aggregator_weights, tiny_flow_config, tiny_head_weights};
use super::prompt::MingTokenizer;
use super::speaker::SPEAKER_EMBEDDING_DIM;
use super::stack::tests::{random, tiny_stack_config, tiny_stack_weights};
use super::vae::tests::{tiny_decoder_weights, tiny_encoder_weights};
use super::{MingControl, MingRequest, MingRuntime, MingVoice};
use crate::models::shared::noise::SplitMix64;

pub(crate) fn tiny_config() -> MingTtsConfig {
    let mut cfg = MingTtsConfig::pinned();
    cfg.llm = tiny_stack_config(2);
    cfg.aggregator = tiny_flow_config();
    cfg.flow = tiny_flow_config();
    cfg.vae_stack = tiny_stack_config(2);
    cfg.vae_group_stack = tiny_stack_config(1);
    cfg
}

/// A word-level tokenizer whose special tokens sit at the trained ids;
/// every other word is one of a few low ids.
pub(crate) fn tiny_tokenizer() -> MingTokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("<unk>".to_string(), 0u32);
    for (index, word) in [
        "hello",
        "there",
        "friend",
        "one",
        "two",
        "three",
        "first",
        "row",
        "here",
        "another",
        "and",
        "the",
        "third",
        "of",
        "text",
        "keep",
        "going",
        "system",
        "user",
        "assistant",
    ]
    .iter()
    .enumerate()
    {
        vocab.insert(word.to_string(), index as u32 + 1);
    }
    for (token, id) in PINNED_TOKENS {
        vocab.insert(token.to_string(), id);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    let specials = PINNED_TOKENS
        .iter()
        .map(|(token, _)| AddedToken::from(*token, true))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(specials).unwrap();
    MingTokenizer::from_tokenizer(tokenizer)
}

pub(crate) fn tiny_weights(cfg: &MingTtsConfig) -> HashMap<String, Tensor> {
    let mut rng = SplitMix64::new(0x4d49_4e47);
    let mut tensors = HashMap::new();
    let hidden = cfg.llm.hidden_size;
    tensors.insert(
        "model.model.embed_tokens.weight".to_string(),
        random(&mut rng, &[cfg.vocab_size, hidden], 0.5),
    );
    tiny_stack_weights(&cfg.llm, "model.model.", &mut rng, &mut tensors);
    tiny_aggregator_weights(
        &cfg.aggregator,
        "linear_proj_audio.",
        hidden,
        &mut rng,
        &mut tensors,
    );
    tiny_head_weights(
        &cfg.flow,
        "flowloss.cfm.model.",
        hidden,
        &mut rng,
        &mut tensors,
    );
    tensors.insert(
        "stop_head.weight".into(),
        random(&mut rng, &[2, hidden], 0.3),
    );
    tensors.insert("stop_head.bias".into(), random(&mut rng, &[2], 0.1));
    tensors.insert(
        "spk_head.weight".into(),
        random(&mut rng, &[hidden, SPEAKER_EMBEDDING_DIM], 0.1),
    );
    tensors.insert("spk_head.bias".into(), random(&mut rng, &[hidden], 0.1));
    tiny_decoder_weights(cfg, "audio.decoder.", &mut rng, &mut tensors);
    tiny_encoder_weights(cfg, "audio.encoder.", &mut rng, &mut tensors);
    tensors
}

/// The tiny model on the CPU in F32, built once.
pub(crate) fn tiny_runtime() -> Arc<MingRuntime> {
    static RUNTIME: OnceLock<Arc<MingRuntime>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| Arc::new(tiny_runtime_on(&Device::Cpu, DType::F32)))
        .clone()
}

pub(crate) fn tiny_runtime_on(device: &Device, dtype: DType) -> MingRuntime {
    let cfg = tiny_config();
    let tensors = tiny_weights(&cfg);
    let lm = VarBuilder::from_tensors(tensors.clone(), dtype, device);
    let vae = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
    let host = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    MingRuntime::assemble(
        Path::new("synthetic"),
        cfg,
        tiny_tokenizer(),
        &lm,
        &vae,
        &host,
        None,
        device,
    )
    .unwrap()
}

/// A built-in-voice request for `text`.
pub(crate) fn tiny_request(text: &str) -> MingRequest {
    MingRequest {
        text: text.to_string(),
        language: None,
        voice: MingVoice::Captioned,
        control: MingControl::BuiltInVoice("US_Emma".into()),
    }
}

//! A tiny checkpoint in Rumble's published layout, for tests: a config, a
//! preprocessor config, a byte-level tokenizer with Rumble's added tokens and
//! one `model.safetensors` holding every tensor the plan names. The weights
//! are deterministic noise, so transcripts are arbitrary; the tests check the
//! loader, the prompt, the decode loop and its bookkeeping.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use crate::kernels::numeric_check::xorshift_values;

use super::checkpoint::{rumble_tensor_plan, RUMBLE_WEIGHTS_FILE};
use super::config::RumbleConfig;

/// Added tokens after the 96 byte-level tokens, as (content, special).
const ADDED_TOKENS: &[(&str, bool)] = &[
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<|audio_start|>", true),
    ("<|audio_end|>", true),
    ("<|audio_pad|>", true),
    ("<non_speech>", false),
    ("<asr_text>", false),
    ("<digits>", true),
    ("<nodigits>", true),
    ("<filler>", true),
    ("<nofiller>", true),
    ("<verbatim>", true),
    ("<noverbatim>", true),
];
const BYTE_TOKENS: usize = 96;
pub(crate) const SYNTHETIC_VOCAB: usize = BYTE_TOKENS + ADDED_TOKENS.len();

pub(crate) fn synthetic_token_id(content: &str) -> u32 {
    let index = ADDED_TOKENS
        .iter()
        .position(|(token, _)| *token == content)
        .expect("synthetic added token");
    (BYTE_TOKENS + index) as u32
}

pub(crate) fn synthetic_config_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "audio_config": {
            "activation_function": "gelu",
            "conv_chunksize": 500,
            "d_model": 16,
            "downsample_hidden_size": 4,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "encoder_layers": 1,
            "n_window": 50,
            "n_window_infer": 800,
            "num_mel_bins": 8,
            "output_dim": 16,
        },
        "audio_start_token_id": synthetic_token_id("<|audio_start|>"),
        "audio_end_token_id": synthetic_token_id("<|audio_end|>"),
        "audio_token_id": synthetic_token_id("<|audio_pad|>"),
        "dtype": "bfloat16",
        "eos_token_id": [
            synthetic_token_id("<|endoftext|>"),
            synthetic_token_id("<|im_end|>"),
        ],
        "pad_token_id": synthetic_token_id("<|endoftext|>"),
        "support_languages": ["English", "Dutch"],
        "text_config": {
            "head_dim": 8,
            "hidden_act": "silu",
            "hidden_size": 16,
            "intermediate_size": 32,
            "max_position_embeddings": 512,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 1_000_000, "rope_type": "default" },
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "vocab_size": SYNTHETIC_VOCAB,
        },
        "tie_word_embeddings": true,
    })
}

fn synthetic_tokenizer_json() -> serde_json::Value {
    // Byte-level symbols of printable ASCII, then space and newline.
    let mut vocab = serde_json::Map::new();
    for (index, byte) in (33_u8..=126).enumerate() {
        vocab.insert((byte as char).to_string(), index.into());
    }
    vocab.insert("\u{0120}".into(), 94.into());
    vocab.insert("\u{010A}".into(), 95.into());
    assert_eq!(vocab.len(), BYTE_TOKENS);
    let added = ADDED_TOKENS
        .iter()
        .enumerate()
        .map(|(index, (content, special))| {
            serde_json::json!({
                "id": BYTE_TOKENS + index,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": special,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true,
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": [],
        },
    })
}

/// Deterministic weights for every tensor of the plan: norm gains near one,
/// everything else small noise.
pub(crate) fn synthetic_tensors(config: &RumbleConfig) -> HashMap<String, Tensor> {
    let device = Device::Cpu;
    rumble_tensor_plan(config)
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let elements = shape.iter().product::<usize>();
            let is_gain = name.ends_with("norm.weight")
                || name.ends_with("layernorm.weight")
                || name.ends_with("ln_post.weight");
            // Large enough that activations clear the norms' epsilons, so
            // the audio visibly conditions the output.
            let noise = xorshift_values(elements, 0x5EED + index as u64, 0.4);
            let values = if is_gain {
                noise.into_iter().map(|value| 1.0 + value * 0.2).collect()
            } else {
                noise
            };
            let tensor = Tensor::from_vec(values, shape, &device).expect("synthetic tensor");
            (name, tensor)
        })
        .collect()
}

/// A temporary directory holding one synthetic checkpoint; removed on drop.
pub(crate) struct SyntheticRumbleCheckpoint {
    dir: PathBuf,
}

impl SyntheticRumbleCheckpoint {
    pub(crate) fn write() -> Self {
        Self::write_with(|_| {})
    }

    /// Writes the checkpoint after `edit` has changed its tensors.
    pub(crate) fn write_with(edit: impl FnOnce(&mut HashMap<String, Tensor>)) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "izwi-rumble-synthetic-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("synthetic dir");
        let config_json = synthetic_config_json();
        std::fs::write(dir.join("config.json"), config_json.to_string()).expect("config");
        std::fs::write(
            dir.join("preprocessor_config.json"),
            serde_json::json!({
                "feature_size": 8,
                "hop_length": 160,
                "min_length": 8000,
                "n_fft": 400,
                "n_window": 50,
                "sampling_rate": 16000,
            })
            .to_string(),
        )
        .expect("preprocessor config");
        std::fs::write(
            dir.join("tokenizer.json"),
            synthetic_tokenizer_json().to_string(),
        )
        .expect("tokenizer");
        let config = RumbleConfig::from_json(&config_json.to_string()).expect("synthetic config");
        let mut tensors = synthetic_tensors(&config);
        edit(&mut tensors);
        candle_core::safetensors::save(&tensors, dir.join(RUMBLE_WEIGHTS_FILE))
            .expect("synthetic weights");
        Self { dir }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for SyntheticRumbleCheckpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

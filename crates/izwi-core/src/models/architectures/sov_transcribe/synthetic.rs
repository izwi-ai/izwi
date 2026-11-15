//! A tiny checkpoint in the published layout, for tests: every real tensor
//! name and dtype (BF16 weights, F32 routers, the stored tied head), two
//! shards behind an index, configs that pass the same validation, and a
//! word-level tokenizer with the chat and audio control tokens.
//!
//! Geometry: an 8-bin, 24-frame window (12 encoder positions, 3 audio
//! tokens); a 2-layer encoder of width 16; a decoder of hidden 32 with
//! layers `sliding full sliding full`, 4 query heads, sliding 2 KV heads of
//! 8 dims with a 6-token window, full 1 KV head of 16 dims rotating 2 of
//! its 8 pairs; a dense MLP of 48 and 6 experts (top-2) of width 12.

use std::collections::BTreeMap;
use std::path::Path;

use half::bf16;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::json;

use super::checkpoint::{expected_census, shard_files, EMBEDDING, INDEX_FILE, LM_HEAD};
use super::config::SovTranscribeConfig;
use crate::kernels::numeric_check::xorshift_values;

pub const HIDDEN: usize = 32;
pub const VOCAB: usize = 96;
pub const AUDIO_TOKEN: u32 = 10;
pub const EOS: u32 = 1;
pub const TURN_END: u32 = 6;
pub const SHARDS: usize = 2;
pub const WINDOW_SAMPLES: usize = 3_840;
pub const TOKENS_PER_WINDOW: usize = 3;
/// The first plain word id.
pub const FIRST_WORD: u32 = 16;

/// Encoder positions of the tiny window the family's own tests use.
pub const POSITIONS: usize = 12;
/// Encoder positions of a published 30 s window, for tests that run real
/// speech through the runtime (375 audio tokens a window, as published).
pub const PUBLISHED_POSITIONS: usize = 1_500;

pub fn config_json() -> serde_json::Value {
    config_json_with(POSITIONS)
}

/// [`config_json`] with windows of `positions` encoder frames.
pub fn config_json_with(positions: usize) -> serde_json::Value {
    json!({
        "architectures": ["SovereignAITranscribeForConditionalGeneration"],
        "model_type": "sovereignai_transcribe",
        "router_dtype": "float32",
        "downsample_factor": 4,
        "audio_token_id": AUDIO_TOKEN,
        "bos_token_id": 2,
        "eos_token_id": EOS,
        "pad_token_id": 0,
        "audio_config": {
            "model_type": "whisper",
            "activation_function": "gelu",
            "scale_embedding": false,
            "num_mel_bins": 8,
            "d_model": 16,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "max_source_positions": positions
        },
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "enable_moe_block": true,
            "attention_k_eq_v": true,
            "use_double_wide_mlp": false,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "attention_bias": false,
            "tie_word_embeddings": true,
            "use_bidirectional_attention": "vision",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 16,
            "sliding_window": 6,
            "layer_types": [
                "sliding_attention", "full_attention", "sliding_attention", "full_attention"
            ],
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                }
            },
            "intermediate_size": 48,
            "num_experts": 6,
            "top_k_experts": 2,
            "moe_intermediate_size": 12,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "vocab_size": VOCAB
        }
    })
}

pub fn preprocessor_json() -> serde_json::Value {
    preprocessor_json_with(POSITIONS)
}

pub fn preprocessor_json_with(positions: usize) -> serde_json::Value {
    json!({
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 8,
        "sampling_rate": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "n_samples": positions * 2 * 160,
        "nb_max_frames": positions * 2,
        "padding_value": 0.0,
        "dither": 0.0
    })
}

/// The word-level tokenizer: control tokens at fixed ids, words after.
pub fn tokenizer_json() -> serde_json::Value {
    let specials = [
        (0, "<pad>"),
        (EOS, "<eos>"),
        (2, "<bos>"),
        (3, "<unk>"),
        (4, "<mask>"),
        (5, "<|turn>"),
        (TURN_END, "<turn|>"),
        (7, "<|channel>"),
        (8, "<channel|>"),
        (9, "<|audio>"),
        (AUDIO_TOKEN, "<|audio|>"),
        (11, "<audio|>"),
        (12, "<|tool_response>"),
    ];
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (id, content) in specials {
        vocab.insert(content.to_string(), json!(id));
        added.push(
            json!({"id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}),
        );
    }
    for (id, word) in [(13, "user"), (14, "model"), (15, "thought")] {
        vocab.insert(word.to_string(), json!(id));
    }
    for id in FIRST_WORD..VOCAB as u32 {
        vocab.insert(format!("w{id}"), json!(id));
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
}

/// Writes the synthetic checkpoint into `dir`.
pub fn write_checkpoint(dir: &Path) {
    write_checkpoint_with(dir, POSITIONS);
}

/// [`write_checkpoint`] with windows of `positions` encoder frames.
pub fn write_checkpoint_with(dir: &Path, positions: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let config = SovTranscribeConfig::parse(
        &config_json_with(positions),
        &preprocessor_json_with(positions),
        &generation_json(),
    )
    .unwrap();
    let files = shard_files(SHARDS);
    let mut shards: Vec<Vec<(String, Dtype, Vec<usize>, Vec<u8>)>> = vec![Vec::new(); SHARDS];
    let mut weight_map = BTreeMap::new();
    // The stored head repeats the embedding (the census lists it first).
    let embedding = xorshift_values(VOCAB * HIDDEN, 99, 0.3);
    for (index, entry) in expected_census(&config).into_iter().enumerate() {
        let len = entry.shape.iter().product::<usize>();
        let name = entry.name.as_str();
        let seed = 100 + index as u64;
        let values = if name == LM_HEAD || name == EMBEDDING {
            embedding.clone()
        } else if name.ends_with("norm.weight")
            || name.ends_with("layernorm.weight")
            || name.ends_with("layernorm_1.weight")
            || name.ends_with("layernorm_2.weight")
            || name.ends_with("layer_norm.weight")
            || name.ends_with("router.scale")
            || name.ends_with("per_expert_scale")
        {
            xorshift_values(len, seed, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect()
        } else if name.ends_with("layer_scalar") {
            vec![0.875]
        } else if name.ends_with(".bias") {
            xorshift_values(len, seed, 0.05)
        } else if name.ends_with("embed_positions.weight") {
            xorshift_values(len, seed, 0.1)
        } else {
            let fan_in = *entry.shape.last().unwrap() as f32;
            xorshift_values(len, seed, 1.5 * fan_in.sqrt().recip())
        };
        let (dtype, bytes) = if entry.f32 {
            (
                Dtype::F32,
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        } else {
            (
                Dtype::BF16,
                values
                    .iter()
                    .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
                    .collect(),
            )
        };
        // The decoder's later layers go to the second shard.
        let shard = usize::from(
            name.contains("language_model.layers.2") || name.contains("language_model.layers.3"),
        );
        weight_map.insert(entry.name.clone(), files[shard].clone());
        shards[shard].push((entry.name, dtype, entry.shape, bytes));
    }
    for (file, tensors) in files.iter().zip(&shards) {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, None, &dir.join(file)).unwrap();
    }
    let write = |file: &str, value: &serde_json::Value| {
        std::fs::write(dir.join(file), serde_json::to_vec_pretty(value).unwrap()).unwrap();
    };
    write(
        INDEX_FILE,
        &json!({"metadata": {}, "weight_map": weight_map}),
    );
    write("config.json", &config_json_with(positions));
    write(
        "preprocessor_config.json",
        &preprocessor_json_with(positions),
    );
    write("generation_config.json", &generation_json());
    write("tokenizer.json", &tokenizer_json());
    write(
        "tokenizer_config.json",
        &json!({"audio_token": "<|audio|>", "eos_token": "<eos>"}),
    );
    std::fs::write(dir.join("chat_template.jinja"), b"{# synthetic #}").unwrap();
    std::fs::write(dir.join(super::precision::SYNTHETIC_MARKER), b"").unwrap();
}

pub fn generation_json() -> serde_json::Value {
    json!({"eos_token_id": [EOS, TURN_END, 12], "pad_token_id": 0, "do_sample": false})
}

/// A directory removed when dropped.
pub struct TempCheckpoint(std::path::PathBuf);

impl TempCheckpoint {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempCheckpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A synthetic checkpoint in a fresh temporary directory.
pub fn temp_checkpoint() -> TempCheckpoint {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "izwi-sov-transcribe-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    write_checkpoint(&dir);
    TempCheckpoint(dir)
}

/// Speech-like test audio: a few harmonics with a slow envelope and noise.
pub fn test_audio(samples: usize, seed: u64) -> Vec<f32> {
    let noise = xorshift_values(samples, seed, 0.01);
    (0..samples)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 9.0 * t).sin();
            let f0 = 140.0 + 30.0 * (seed % 5) as f32;
            envelope
                * (0.3 * (2.0 * std::f32::consts::PI * f0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin())
                + noise[i]
        })
        .collect()
}

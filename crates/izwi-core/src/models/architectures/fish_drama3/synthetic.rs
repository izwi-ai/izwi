//! Synthetic checkpoints for tests: the real tensor names, BF16 dense
//! tensors, E4M3 routed experts with F32 128×128 block scales, a sharded
//! index, `config.json` and the companion files.
//!
//! [`SyntheticCheckpoint`] writes the §10.1 geometry with values shaped like
//! a trained model's (zero-centred norm weights near 0, the DeltaNet norm
//! near 1, `A_log` and `dt_bias` near their published means, projections at
//! `1 / sqrt(fan_in)`), so every convention moves the numbers visibly:
//! * hidden 128 (one FP8 block); 8 layers, full attention at 3 and 7;
//! * attention 4 query / 2 KV heads of 32, rotary 8 (the published 1/4);
//! * DeltaNet 2 key / 4 value heads of 32, kernel 4;
//! * 8 experts, top 2, intermediate 256 (two FP8 row blocks, so the scale
//!   grid is 2×1 and 1×2) and a shared expert of 64;
//! * a 600-token vocabulary, a 64-code semantic band at 512..=575 and
//!   64-entry codebooks; the tower at width 128, 4 + 9 layers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use half::bf16;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::Value;

use super::checkpoint::{
    expected_census, CensusDtype, FishDrama3Checkpoint, COMPANION_FILES, INDEX_FILE, TRUNK_PREFIX,
};
use super::config::tests::PUBLISHED_CONFIG;
use super::config::{FishDrama3Config, FishDrama3ConfigPolicy};
use crate::kernels::cuda::fp8::encode_e4m3fn;
use crate::kernels::numeric_check::xorshift_values;

/// A tensor to write: name, dtype, shape, bytes.
pub(crate) type Written = (String, Dtype, Vec<usize>, Vec<u8>);

/// A directory removed when dropped.
pub(crate) struct TempDir(pub(crate) PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn temp_dir() -> TempDir {
    let dir = std::env::temp_dir().join(format!("izwi-fish-drama3-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

/// Two shards (the trunk, then the head and the tower), the index, `config`
/// as `config.json`, and the companion files.
pub(crate) fn write_checkpoint(dir: &Path, tensors: &[Written], config: &Value) {
    let mut shards = BTreeMap::<&str, Vec<&Written>>::new();
    for tensor in tensors {
        let shard = if tensor.0.starts_with(TRUNK_PREFIX) {
            "model-00001-of-00002.safetensors"
        } else {
            "model-00002-of-00002.safetensors"
        };
        shards.entry(shard).or_default().push(tensor);
    }
    let mut weight_map = BTreeMap::new();
    let mut total = 0usize;
    for (shard, tensors) in &shards {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                weight_map.insert(name.clone(), shard.to_string());
                total += bytes.len();
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, None, &dir.join(shard)).unwrap();
    }
    std::fs::write(
        dir.join(INDEX_FILE),
        serde_json::to_vec_pretty(&serde_json::json!({
            "metadata": {"total_size": total},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    for name in COMPANION_FILES {
        std::fs::write(dir.join(name), b"synthetic").unwrap();
    }
}

/// Deterministic byte patterns for every census entry (the census and
/// reader tests): BF16 values in [-0.5, 0.5), positive F32 scales, E4M3
/// codes without the NaN.
pub(crate) fn pattern_tensors(config: &FishDrama3Config) -> Vec<Written> {
    expected_census(config)
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            let n = e.shape.iter().product::<usize>();
            let bytes = match e.dtype {
                CensusDtype::Bf16 => (0..n)
                    .flat_map(|k| {
                        let v = ((i * 131 + k * 17) % 251) as f32 / 251.0 - 0.5;
                        bf16::from_f32(v).to_le_bytes()
                    })
                    .collect(),
                CensusDtype::F32 => (0..n)
                    .flat_map(|k| (1e-3 * (1 + (i + k) % 7) as f32).to_le_bytes())
                    .collect(),
                CensusDtype::F8E4M3 => (0..n)
                    .map(|k| {
                        let code = ((i * 31 + k * 7) % 256) as u8;
                        if code & 0x7f == 0x7f {
                            code ^ 1
                        } else {
                            code
                        }
                    })
                    .collect(),
            };
            (e.name, e.dtype.safetensors(), e.shape, bytes)
        })
        .collect()
}

/// The §10.1 geometry, every switch as published.
pub(crate) fn trunk_geometry_json() -> Value {
    let mut c: Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
    let t = &mut c["text_config"];
    t["hidden_size"] = 128.into();
    t["num_hidden_layers"] = 8.into();
    let kinds = (0..8)
        .map(|layer| {
            if layer % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    t["layer_types"] = serde_json::json!(kinds);
    t["mlp_layer_types"] = serde_json::json!(vec!["moe"; 8]);
    t["num_attention_heads"] = 4.into();
    t["num_key_value_heads"] = 2.into();
    t["head_dim"] = 32.into();
    t["linear_num_key_heads"] = 2.into();
    t["linear_key_head_dim"] = 32.into();
    t["linear_num_value_heads"] = 4.into();
    t["linear_value_head_dim"] = 32.into();
    t["num_experts"] = 8.into();
    t["num_experts_per_tok"] = 2.into();
    t["moe_intermediate_size"] = 256.into();
    t["shared_expert_intermediate_size"] = 64.into();
    t["vocab_size"] = 600.into();
    t["max_position_embeddings"] = 4096.into();
    t["eos_token_id"] = 470.into();
    t["pad_token_id"] = 468.into();
    let a = &mut c["audio_decoder_config"];
    a["dim"] = 128.into();
    a["text_dim"] = 128.into();
    a["n_head"] = 4.into();
    a["n_local_heads"] = 2.into();
    a["head_dim"] = 32.into();
    a["intermediate_size"] = 192.into();
    a["vocab_size"] = 64.into();
    c["eos_token_id"] = 470.into();
    c["pad_token_id"] = 468.into();
    c["audio_pad_token_id"] = 480.into();
    c["semantic_start_token_id"] = 512.into();
    c["semantic_end_token_id"] = 575.into();
    c
}

pub(crate) fn trunk_geometry() -> FishDrama3Config {
    FishDrama3Config::from_json(
        &trunk_geometry_json().to_string(),
        FishDrama3ConfigPolicy::Synthetic,
    )
    .unwrap()
}

/// FNV-1a: a stable per-tensor seed.
fn name_seed(name: &str, seed: u64) -> u64 {
    name.bytes()
        .fold(0xcbf2_9ce4_8422_2325 ^ seed, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// A BF16 tensor's values by its role.
fn dense_values(name: &str, shape: &[usize], seed: u64) -> Vec<f32> {
    let n = shape.iter().product::<usize>();
    let cols = *shape.last().unwrap();
    let draw = |scale: f32| xorshift_values(n, name_seed(name, seed), scale);
    let around = |center: f32, spread: f32| draw(spread).into_iter().map(|v| v + center).collect();
    if name.ends_with("linear_attn.norm.weight") {
        // The DeltaNet output norm's plain gain.
        around(1.0, 0.2)
    } else if name.ends_with("norm.weight") {
        // Zero-centred gains (input, post-attention, final, q/k, tower).
        draw(0.2)
    } else if name.ends_with("A_log") {
        around(-1.5, 0.8)
    } else if name.ends_with("dt_bias") {
        around(-0.8, 0.4)
    } else if name.ends_with("conv1d.weight") {
        draw(0.5)
    } else if name.ends_with("embed_tokens.weight") || name.ends_with("embedding.weight") {
        draw(1.0)
    } else if name.ends_with("codebook_embeddings.weight") {
        draw(0.5)
    } else if name.ends_with("mlp.gate.weight") {
        // A sharper router, so the top-k is not a coin toss.
        draw(2.0 * (3.0 / cols as f32).sqrt())
    } else {
        // Projections at variance 1 / fan_in.
        draw((3.0 / cols as f32).sqrt())
    }
}

/// An E4M3 expert projection and its F32 scales: per 128×128 block,
/// `scale = amax / 448` (times a per-block factor, so scales differ).
fn fp8_values(
    name: &str,
    rows: usize,
    cols: usize,
    block: usize,
    seed: u64,
) -> (Vec<u8>, Vec<f32>) {
    let values = xorshift_values(
        rows * cols,
        name_seed(name, seed),
        (3.0 / cols as f32).sqrt(),
    );
    let (row_blocks, col_blocks) = (rows.div_ceil(block), cols.div_ceil(block));
    let mut scales = vec![0f32; row_blocks * col_blocks];
    for (b, scale) in scales.iter_mut().enumerate() {
        let (rb, cb) = (b / col_blocks, b % col_blocks);
        let mut amax = 0f32;
        for r in rb * block..((rb + 1) * block).min(rows) {
            for c in cb * block..((cb + 1) * block).min(cols) {
                amax = amax.max(values[r * cols + c].abs());
            }
        }
        *scale = amax / 448.0 * (1.0 + 0.25 * b as f32);
    }
    let codes = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (r, c) = (i / cols, i % cols);
            encode_e4m3fn(v / scales[(r / block) * col_blocks + c / block])
        })
        .collect();
    (codes, scales)
}

/// Values for every census entry of `config`, from `seed`.
pub(crate) fn trained_like_tensors(config: &FishDrama3Config, seed: u64) -> Vec<Written> {
    let census = expected_census(config);
    let block = config.fp8_block;
    let mut scales = BTreeMap::new();
    census
        .into_iter()
        .map(|e| {
            let bytes = match e.dtype {
                CensusDtype::Bf16 => dense_values(&e.name, &e.shape, seed)
                    .into_iter()
                    .flat_map(|v| bf16::from_f32(v).to_le_bytes())
                    .collect(),
                CensusDtype::F8E4M3 => {
                    let (codes, block_scales) =
                        fp8_values(&e.name, e.shape[0], e.shape[1], block, seed);
                    let module = e.name.trim_end_matches(".weight").to_string();
                    scales.insert(module, block_scales);
                    codes
                }
                CensusDtype::F32 => {
                    // The census lists a weight before its scales.
                    let module = e.name.trim_end_matches(".weight_scale_inv");
                    scales
                        .remove(module)
                        .expect("an FP8 weight precedes its scales")
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect()
                }
            };
            (e.name, e.dtype.safetensors(), e.shape, bytes)
        })
        .collect()
}

/// A written §10.1 checkpoint, removed when dropped.
pub(crate) struct SyntheticCheckpoint {
    pub(crate) dir: TempDir,
    pub(crate) config: FishDrama3Config,
}

impl SyntheticCheckpoint {
    pub(crate) fn write(seed: u64) -> Self {
        Self::write_with(seed, |_| {})
    }

    /// The checkpoint with one edit to its tensors.
    pub(crate) fn write_with(seed: u64, edit: impl FnOnce(&mut Vec<Written>)) -> Self {
        let config = trunk_geometry();
        let mut tensors = trained_like_tensors(&config, seed);
        edit(&mut tensors);
        let dir = temp_dir();
        write_checkpoint(&dir.0, &tensors, &trunk_geometry_json());
        Self { dir, config }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.dir.0
    }

    /// The checkpoint, opened and census-checked.
    pub(crate) fn open(&self) -> FishDrama3Checkpoint {
        let checkpoint = FishDrama3Checkpoint::open(self.path()).unwrap();
        checkpoint.validate_census(&self.config).unwrap();
        checkpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::fp8::decode_e4m3fn;

    #[test]
    fn the_synthetic_geometry_loads_and_passes_its_census() {
        let synthetic = SyntheticCheckpoint::write(0);
        let c = &synthetic.config;
        assert_eq!(c.policy, FishDrama3ConfigPolicy::Synthetic);
        assert_eq!((c.trunk.hidden, c.trunk.layers.len()), (128, 8));
        assert_eq!(c.trunk.full_attention_layers(), vec![3, 7]);
        assert_eq!(c.trunk.attention.rotary_dims, 8);
        assert_eq!(c.trunk.linear.conv_channels(), 256);
        assert_eq!((c.trunk.moe.experts, c.trunk.moe.top_k), (8, 2));
        assert_eq!(c.ids.semantic_codes(), 64);
        assert_eq!(c.ids.compact_head_rows().len(), 65);
        assert_eq!(c.tower.residual_layers(), 9);
        let config =
            FishDrama3Config::load_with_policy(synthetic.path(), FishDrama3ConfigPolicy::Synthetic)
                .unwrap();
        assert_eq!(&config, c);
        let checkpoint = synthetic.open();
        assert_eq!(checkpoint.tensor_count(), expected_census(c).len());
    }

    /// The written experts are E4M3 codes whose dequantization recovers the
    /// drawn values: the scale is multiplied, never divided.
    #[test]
    fn the_written_experts_dequantize_by_multiplying_their_block_scales() {
        let synthetic = SyntheticCheckpoint::write(0);
        let checkpoint = synthetic.open();
        let module = format!("{}.gate_proj", super::super::checkpoint::expert(2, 5));
        let weight = checkpoint.fp8_block_weight(&module, 256, 128, 128).unwrap();
        assert_eq!(weight.scale_shape(), [2, 1]);
        assert_ne!(weight.scales[0], weight.scales[1]);
        let drawn = xorshift_values(
            256 * 128,
            name_seed(&format!("{module}.weight"), 0),
            (3.0f32 / 128.0).sqrt(),
        );
        let dequantized = weight.dequantized();
        for (i, (&code, value)) in weight.codes.iter().zip(&dequantized).enumerate() {
            let scale = weight.scales[i / 128 / 128];
            assert_eq!(*value, decode_e4m3fn(code) * scale);
        }
        // E4M3 keeps 3 mantissa bits: within 1/16 of each block's scale grid.
        let error = crate::kernels::numeric_check::relative_l2(&dequantized, &drawn);
        assert!(error < 0.05, "{error}");
        let divided = weight
            .codes
            .iter()
            .enumerate()
            .map(|(i, &code)| decode_e4m3fn(code) / weight.scales[i / 128 / 128])
            .collect::<Vec<_>>();
        assert!(crate::kernels::numeric_check::relative_l2(&divided, &drawn) > 100.0);
    }
}

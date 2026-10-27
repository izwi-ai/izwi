//! Maya1's decoder from a GGUF release: llama.cpp's `llama` conversion of
//! the pinned checkpoint, quantized to Q8_0 (`mradermacher/maya1-GGUF`).
//!
//! The projections, the token table and the head slice stay quantized
//! (`QMatMul`, the shared quantized token table); activations and the KV
//! cache run in F32. The conversion reorders each head's Q and K rows for
//! interleaved RoPE, which the loader puts back to the half-split order the
//! trunk uses, and stores Llama3's RoPE scaling as `rope_freqs` rather than
//! metadata, which the loader checks against the pinned config's.

use std::path::Path;

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device};
use candle_nn::RmsNorm;

use super::config::{Maya1LmConfig, HEAD_FIRST_TOKEN, HEAD_ROWS};
use super::trunk::{Maya1Layer, Maya1Trunk, Projection, TokenTable};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::models::shared::rope::{rope_frequencies, RopeScaling};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::gguf_projection::{
    joined_qmatmul, row_slice_qtensor, QuantizedPart,
};
use crate::models::shared::weights::token_table::QuantizedTokenTable;

/// The quantized checkpoint the GGUF variant downloads.
pub const MAYA1_GGUF_FILE: &str = "maya1.Q8_0.gguf";

/// Load the decoder from `path` onto `device`: weights quantized as stored,
/// activations in F32.
pub(crate) fn load_gguf_trunk(
    path: &Path,
    cfg: Maya1LmConfig,
    device: &Device,
    max_positions: usize,
) -> Result<Maya1Trunk> {
    if !path.is_file() {
        return Err(Error::ModelLoadError(format!(
            "Maya1 GGUF weights {} are missing",
            path.display()
        )));
    }
    let gguf = GgufLoader::from_path_with_backend(path, BackendKind::Cpu)?;
    check_metadata(&gguf, &cfg)?;
    check_rope_freqs(&gguf, &cfg)?;
    let table = QuantizedTokenTable::new(gguf.load_qtensor("token_embd.weight", device)?)?;
    let head = QMatMul::from_qtensor(row_slice_qtensor(
        &gguf,
        "token_embd.weight",
        HEAD_FIRST_TOKEN as usize..HEAD_FIRST_TOKEN as usize + HEAD_ROWS,
        device,
    )?)?;
    let eps = cfg.rms_norm_eps;
    let norm = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm::new(
            gguf.load_tensor(name, DType::F32, device)?,
            eps,
        ))
    };
    let layers = (0..cfg.num_hidden_layers)
        .map(|layer| -> Result<Maya1Layer> {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}.weight");
            let (q, k, v) = (name("attn_q"), name("attn_k"), name("attn_v"));
            let (gate, up) = (name("ffn_gate"), name("ffn_up"));
            let single = |tensor: &str| -> Result<Projection> {
                Ok(Projection::Quantized(joined_qmatmul(
                    &gguf,
                    &[QuantizedPart {
                        name: tensor,
                        rope_heads: None,
                    }],
                    device,
                )?))
            };
            Ok(Maya1Layer {
                input_norm: norm(&name("attn_norm"))?,
                qkv: Projection::Quantized(joined_qmatmul(
                    &gguf,
                    &[
                        QuantizedPart {
                            name: &q,
                            rope_heads: Some(cfg.num_attention_heads),
                        },
                        QuantizedPart {
                            name: &k,
                            rope_heads: Some(cfg.num_key_value_heads),
                        },
                        QuantizedPart {
                            name: &v,
                            rope_heads: None,
                        },
                    ],
                    device,
                )?),
                o_proj: single(&name("attn_output"))?,
                post_norm: norm(&name("ffn_norm"))?,
                gate_up: Projection::Quantized(joined_qmatmul(
                    &gguf,
                    &[
                        QuantizedPart {
                            name: &gate,
                            rope_heads: None,
                        },
                        QuantizedPart {
                            name: &up,
                            rope_heads: None,
                        },
                    ],
                    device,
                )?),
                down: single(&name("ffn_down"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let output_norm = norm("output_norm.weight")?;
    Maya1Trunk::assemble(
        cfg,
        DType::F32,
        device.clone(),
        TokenTable::Quantized(table),
        Projection::Quantized(head),
        layers,
        output_norm,
        max_positions,
    )
}

/// The GGUF's geometry must be the pinned config's.
fn check_metadata(gguf: &GgufLoader, cfg: &Maya1LmConfig) -> Result<()> {
    let architecture = gguf.get_metadata_string("general.architecture");
    if architecture.as_deref() != Some("llama") {
        return Err(Error::ModelLoadError(format!(
            "Maya1 GGUF is a {architecture:?} model, not llama"
        )));
    }
    for (key, expected) in [
        ("llama.block_count", cfg.num_hidden_layers),
        ("llama.embedding_length", cfg.hidden_size),
        ("llama.feed_forward_length", cfg.intermediate_size),
        ("llama.attention.head_count", cfg.num_attention_heads),
        ("llama.attention.head_count_kv", cfg.num_key_value_heads),
        ("llama.attention.key_length", cfg.head_dim),
        ("llama.attention.value_length", cfg.head_dim),
        ("llama.vocab_size", cfg.vocab_size),
    ] {
        let found = gguf.get_metadata_u64(key);
        if found != Some(expected as u64) {
            return Err(Error::ModelLoadError(format!(
                "Maya1 GGUF metadata {key} is {found:?}, but the pinned config gives {expected}"
            )));
        }
    }
    for (key, expected) in [
        ("llama.rope.freq_base", cfg.rope_theta),
        ("llama.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps),
    ] {
        let found = gguf.get_metadata_f64(key);
        if !found.is_some_and(|value| (value - expected).abs() <= expected.abs() * 1e-6) {
            return Err(Error::ModelLoadError(format!(
                "Maya1 GGUF metadata {key} is {found:?}, but the pinned config gives {expected}"
            )));
        }
    }
    if gguf.has_tensor("output.weight") {
        return Err(Error::ModelLoadError(
            "Maya1 GGUF has an untied output head; the checkpoint ties it".into(),
        ));
    }
    Ok(())
}

/// llama.cpp's `rope_freqs` (each frequency's divisor) must equal the
/// pinned config's Llama3 scaling.
fn check_rope_freqs(gguf: &GgufLoader, cfg: &Maya1LmConfig) -> Result<()> {
    let stored = gguf
        .load_tensor("rope_freqs.weight", DType::F32, &Device::Cpu)?
        .to_vec1::<f32>()?;
    let unscaled = rope_frequencies(cfg.head_dim, cfg.rope_theta, &RopeScaling::Default)?;
    let scaled = rope_frequencies(cfg.head_dim, cfg.rope_theta, &cfg.rope_scaling)?;
    if stored.len() != scaled.inv_freq.len() {
        return Err(Error::ModelLoadError(format!(
            "Maya1 GGUF rope_freqs has {} factors for {} frequencies",
            stored.len(),
            scaled.inv_freq.len()
        )));
    }
    for (index, ((&factor, base), ours)) in stored
        .iter()
        .zip(&unscaled.inv_freq)
        .zip(&scaled.inv_freq)
        .enumerate()
    {
        let expected = base / ours;
        if (f64::from(factor) - expected).abs() > expected * 1e-5 {
            return Err(Error::ModelLoadError(format!(
                "Maya1 GGUF rope_freqs[{index}] is {factor}, but Llama3 scaling gives {expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::Tensor;

    use super::*;
    use crate::models::architectures::maya1::trunk::tests::{
        batch_variance, dense_reference, head_128_config, max_diff, tiny_config, tiny_weights,
    };
    use crate::models::architectures::maya1::trunk::Maya1ForwardRow;
    use crate::models::shared::attention::physical::PhysicalPagedKvCache;

    /// llama.cpp's `permute`: each head's half-split rows interleaved.
    fn interleave(weight: &Tensor, heads: usize) -> Tensor {
        let (rows, columns) = weight.dims2().unwrap();
        weight
            .reshape((heads, 2, rows / heads / 2, columns))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((rows, columns))
            .unwrap()
    }

    /// `cfg`'s random Llama written as llama.cpp's `llama` conversion
    /// writes Maya1 (Q8_0 projections and table, F32 norms, Q and K
    /// interleaved, `rope_freqs`, no output head), and the weights the GGUF
    /// holds in checkpoint order: each Q8_0 tensor dequantized.
    fn write_tiny_gguf(cfg: &Maya1LmConfig, path: &Path) -> HashMap<String, Tensor> {
        let weights = tiny_weights(cfg);
        let mut tensors: Vec<(String, QTensor)> = Vec::new();
        let mut held = HashMap::new();
        let mut quantized = |gguf: String, hf: String, heads: Option<usize>| {
            let weight = &weights[&hf];
            let stored = match heads {
                Some(heads) => interleave(weight, heads),
                None => weight.clone(),
            };
            tensors.push((gguf, QTensor::quantize(&stored, GgmlDType::Q8_0).unwrap()));
            let roundtrip = QTensor::quantize(weight, GgmlDType::Q8_0)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap();
            held.insert(hf, roundtrip);
        };
        quantized(
            "token_embd.weight".into(),
            "model.embed_tokens.weight".into(),
            None,
        );
        for layer in 0..cfg.num_hidden_layers {
            for (gguf, hf, heads) in [
                ("attn_q", "self_attn.q_proj", Some(cfg.num_attention_heads)),
                ("attn_k", "self_attn.k_proj", Some(cfg.num_key_value_heads)),
                ("attn_v", "self_attn.v_proj", None),
                ("attn_output", "self_attn.o_proj", None),
                ("ffn_gate", "mlp.gate_proj", None),
                ("ffn_up", "mlp.up_proj", None),
                ("ffn_down", "mlp.down_proj", None),
            ] {
                quantized(
                    format!("blk.{layer}.{gguf}.weight"),
                    format!("model.layers.{layer}.{hf}.weight"),
                    heads,
                );
            }
        }
        let mut dense = |gguf: String, hf: String| {
            tensors.push((
                gguf,
                QTensor::quantize(&weights[&hf], GgmlDType::F32).unwrap(),
            ));
            held.insert(hf.clone(), weights[&hf].clone());
        };
        dense("output_norm.weight".into(), "model.norm.weight".into());
        for layer in 0..cfg.num_hidden_layers {
            dense(
                format!("blk.{layer}.attn_norm.weight"),
                format!("model.layers.{layer}.input_layernorm.weight"),
            );
            dense(
                format!("blk.{layer}.ffn_norm.weight"),
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
            );
        }
        let unscaled =
            rope_frequencies(cfg.head_dim, cfg.rope_theta, &RopeScaling::Default).unwrap();
        let scaled = rope_frequencies(cfg.head_dim, cfg.rope_theta, &cfg.rope_scaling).unwrap();
        let factors = unscaled
            .inv_freq
            .iter()
            .zip(&scaled.inv_freq)
            .map(|(base, ours)| (base / ours) as f32)
            .collect::<Vec<_>>();
        let factors = Tensor::from_vec(factors, cfg.head_dim / 2, &Device::Cpu).unwrap();
        tensors.push((
            "rope_freqs.weight".into(),
            QTensor::quantize(&factors, GgmlDType::F32).unwrap(),
        ));

        let count = |value: usize| gguf_file::Value::U32(value as u32);
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("llama".into()),
            ),
            ("llama.block_count", count(cfg.num_hidden_layers)),
            ("llama.embedding_length", count(cfg.hidden_size)),
            ("llama.feed_forward_length", count(cfg.intermediate_size)),
            ("llama.attention.head_count", count(cfg.num_attention_heads)),
            (
                "llama.attention.head_count_kv",
                count(cfg.num_key_value_heads),
            ),
            ("llama.attention.key_length", count(cfg.head_dim)),
            ("llama.attention.value_length", count(cfg.head_dim)),
            ("llama.vocab_size", count(cfg.vocab_size)),
            (
                "llama.rope.freq_base",
                gguf_file::Value::F32(cfg.rope_theta as f32),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F32(cfg.rms_norm_eps as f32),
            ),
        ];
        let metadata = metadata
            .iter()
            .map(|(key, value)| (*key, value))
            .collect::<Vec<_>>();
        let tensors = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let mut file = std::fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &metadata, &tensors).unwrap();
        held
    }

    #[test]
    fn gguf_trunk_runs_the_weights_it_stores() {
        let cfg = tiny_config();
        let dir = std::env::temp_dir().join(format!("izwi-maya1-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(MAYA1_GGUF_FILE);
        let held = write_tiny_gguf(&cfg, &path);
        let trunk = load_gguf_trunk(&path, cfg.clone(), &Device::Cpu, 64).unwrap();
        assert_eq!(trunk.dtype(), DType::F32);

        let tokens = [
            128_259u32, 128_000, 40, 7_000, 52_011, 128_009, 128_260, 128_261, 128_257, 128_266,
            132_400, 136_500,
        ];
        let mut cache = PhysicalPagedKvCache::scratch_for_probe(
            &Device::Cpu,
            DType::F32,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            64,
        )
        .unwrap();
        let logits = trunk
            .forward(&mut [Maya1ForwardRow {
                tokens: &tokens,
                cache: &mut cache,
            }])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected = dense_reference(&cfg, &held, &tokens);
        let expected = expected.last().unwrap();
        // CPU Q8_0 matmuls quantize their activations too, so the logits
        // agree to Q8_0's precision, not bit for bit.
        let scale = expected.iter().fold(0f32, |worst, v| worst.max(v.abs()));
        let diff = max_diff(&logits[0], expected);
        assert!(diff < 0.02 * scale, "{diff} against logits up to {scale}");

        // Q and K left interleaved would be a different model: the loader's
        // reorder is what the agreement rests on.
        let mut interleaved = held.clone();
        for layer in 0..cfg.num_hidden_layers {
            for (name, heads) in [
                ("q_proj", cfg.num_attention_heads),
                ("k_proj", cfg.num_key_value_heads),
            ] {
                let key = format!("model.layers.{layer}.self_attn.{name}.weight");
                let weight = interleave(&held[&key], heads);
                interleaved.insert(key, weight);
            }
        }
        let wrong = dense_reference(&cfg, &interleaved, &tokens);
        let wrong_diff = max_diff(&logits[0], wrong.last().unwrap());
        assert!(wrong_diff > 5.0 * diff, "{wrong_diff} vs {diff}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gguf_geometry_must_be_the_pinned_configs() {
        let cfg = tiny_config();
        let dir =
            std::env::temp_dir().join(format!("izwi-maya1-gguf-geometry-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(MAYA1_GGUF_FILE);
        write_tiny_gguf(&cfg, &path);
        let mut other = cfg.clone();
        other.intermediate_size = 128;
        let error = load_gguf_trunk(&path, other, &Device::Cpu, 64)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("llama.feed_forward_length"), "{error}");
        let mut other = cfg.clone();
        other.rope_scaling = RopeScaling::Default;
        let error = load_gguf_trunk(&path, other, &Device::Cpu, 64)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("rope_freqs"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Q8_0 projections give a row the same logits alone or beside other
    /// requests' rows.
    fn quantized_batch_variance(device: &Device, tag: &str) -> f32 {
        let cfg = head_128_config();
        let dir =
            std::env::temp_dir().join(format!("izwi-maya1-gguf-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(MAYA1_GGUF_FILE);
        write_tiny_gguf(&cfg, &path);
        let trunk = load_gguf_trunk(&path, cfg, device, 64).unwrap();
        let variance = batch_variance(&trunk, device, DType::F32);
        std::fs::remove_dir_all(&dir).ok();
        variance
    }

    #[test]
    fn cpu_quantized_rows_are_batch_invariant() {
        assert_eq!(quantized_batch_variance(&Device::Cpu, "cpu-batch"), 0.0);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_quantized_rows_are_batch_invariant() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_quantized_rows_are_batch_invariant")
        else {
            return;
        };
        assert_eq!(quantized_batch_variance(&device, "metal-batch"), 0.0);
    }
}

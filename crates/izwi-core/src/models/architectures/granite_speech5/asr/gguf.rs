//! IBM's llama.cpp GGUF release of Granite Speech 5.0: the language model in
//! one file (`granite` architecture, Q8_0) and the audio encoder and
//! projector in another (`clip` mmproj, F16 projections). The model is the
//! safetensors model; only the container differs, so this module turns both
//! files back into what the 5.0 loaders read:
//!
//! - the mmproj tensors under their safetensors names, read on demand, with
//!   llama.cpp's conversion undone (K and V joined back into `to_kv`, the
//!   squeezed convolution kernels given back their unit dimension, the
//!   folded BatchNorm handed over as scale and shift);
//! - the language model's quantized projections, with llama.cpp's Q/K row
//!   order (interleaved RoPE pairs) put back to the half-split order 5.0's
//!   RoPE uses. Rows of a GGML tensor are whole blocks, so reordering and
//!   joining them never requantizes.
//!
//! Both files' metadata must agree with the release's `config.json`, which
//! the GGUF variant downloads beside them.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::GraniteSpeech5Config;

/// Checks that the two GGUF files describe the model `config` describes:
/// the language model's geometry, YaRN scaling and Granite multipliers, and
/// the encoder's and projector's geometry.
pub(super) fn check_gguf_metadata(
    config: &GraniteSpeech5Config,
    language_model: &GgufLoader,
    mmproj: &GgufLoader,
) -> Result<()> {
    let text = &config.text_config;
    let yarn = text.yarn()?;
    let encoder = &config.encoder_config;
    expect_string(language_model, "general.architecture", "granite")?;
    for (key, expected) in [
        ("granite.block_count", text.num_hidden_layers),
        ("granite.context_length", text.max_position_embeddings),
        ("granite.embedding_length", text.hidden_size),
        ("granite.feed_forward_length", text.intermediate_size),
        ("granite.attention.head_count", text.num_attention_heads),
        ("granite.attention.head_count_kv", text.num_key_value_heads),
        ("granite.vocab_size", text.vocab_size),
        (
            "granite.rope.scaling.original_context_length",
            yarn.original_max_position_embeddings,
        ),
    ] {
        expect_count(language_model, key, expected)?;
    }
    expect_string(language_model, "granite.rope.scaling.type", "yarn")?;
    for (key, expected) in [
        ("granite.rope.scaling.factor", yarn.factor),
        ("granite.rope.freq_base", text.rope_theta),
        (
            "granite.attention.layer_norm_rms_epsilon",
            text.rms_norm_eps,
        ),
        (
            "granite.attention.scale",
            f64::from(text.attention_multiplier),
        ),
        (
            "granite.embedding_scale",
            f64::from(text.embedding_multiplier),
        ),
        (
            "granite.residual_scale",
            f64::from(text.residual_multiplier),
        ),
        ("granite.logit_scale", f64::from(text.logits_scaling)),
    ] {
        expect_value(language_model, key, expected)?;
    }

    expect_string(mmproj, "clip.projector_type", "granite_speech")?;
    for (key, expected) in [
        ("clip.audio.block_count", encoder.num_layers),
        ("clip.audio.embedding_length", encoder.hidden_dim),
        (
            "clip.audio.feed_forward_length",
            encoder.hidden_dim * encoder.feedforward_mult,
        ),
        ("clip.audio.attention.head_count", encoder.num_heads),
        ("clip.audio.num_mel_bins", encoder.input_dim),
        ("clip.audio.chunk_size", encoder.context_size),
        ("clip.audio.conv_kernel_size", encoder.conv_kernel_size),
        ("clip.audio.max_pos_emb", encoder.max_pos_emb),
        ("clip.audio.projection_dim", text.hidden_size),
        ("clip.audio.projector.window_size", config.window_size),
        (
            "clip.audio.projector.downsample_rate",
            config.downsample_rate,
        ),
        (
            "clip.audio.projector.head_count",
            config.projector_config.num_attention_heads,
        ),
    ] {
        expect_count(mmproj, key, expected)?;
    }
    Ok(())
}

fn mismatch(key: &str, found: impl std::fmt::Debug, expected: impl std::fmt::Debug) -> Error {
    Error::ModelLoadError(format!(
        "Granite Speech 5.0 GGUF metadata {key} is {found:?}, but config.json gives {expected:?}"
    ))
}

fn expect_string(gguf: &GgufLoader, key: &str, expected: &str) -> Result<()> {
    let found = gguf.get_metadata_string(key);
    if found.as_deref() == Some(expected) {
        Ok(())
    } else {
        Err(mismatch(key, found, expected))
    }
}

fn expect_count(gguf: &GgufLoader, key: &str, expected: usize) -> Result<()> {
    let found = gguf.get_metadata_u64(key);
    if found == Some(expected as u64) {
        Ok(())
    } else {
        Err(mismatch(key, found, expected))
    }
}

/// GGUF stores these as F32, so the comparison allows F32 rounding.
fn expect_value(gguf: &GgufLoader, key: &str, expected: f64) -> Result<()> {
    let found = gguf.get_metadata_f64(key);
    match found {
        Some(value) if (value - expected).abs() <= expected.abs() * 1e-6 => Ok(()),
        _ => Err(mismatch(key, found, expected)),
    }
}

/// How an mmproj tensor becomes the tensor the safetensors loaders read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MmprojTensor {
    /// The same tensor under a safetensors name.
    Renamed(String),
    /// A squeezed `Conv1d` kernel given back its unit dimension at `dim`.
    Unsqueezed { name: String, dim: usize },
    /// One half of `to_kv`, split by the conversion: `key` comes first.
    KeyValue { name: String, key: bool },
}

/// The safetensors counterpart of mmproj tensor `name`, `None` for a name
/// the conversion does not write.
fn mmproj_tensor(name: &str) -> Option<MmprojTensor> {
    use MmprojTensor::*;
    let renamed = |name: String| Some(Renamed(name));
    let (stem, suffix) = match name.rsplit_once('.') {
        Some((stem, suffix @ ("weight" | "bias"))) => (stem, Some(suffix)),
        _ => (name, None),
    };
    let with_suffix = |base: &str| match suffix {
        Some(suffix) => format!("{base}.{suffix}"),
        None => base.to_string(),
    };
    match stem {
        "a.input_projection" => return renamed(with_suffix("encoder.input_linear")),
        "a.enc_ctc_out" => return renamed(with_suffix("encoder.out")),
        "a.enc_ctc_out_mid" => return renamed(with_suffix("encoder.out_mid")),
        "a.proj_query" if suffix.is_none() => return renamed("projector.query".into()),
        "a.proj_norm" => return renamed(with_suffix("projector.qformer.layernorm")),
        "a.proj_linear" => return renamed(with_suffix("projector.linear")),
        _ => {}
    }
    if let Some((index, part)) = stem
        .strip_prefix("a.blk.")
        .and_then(|rest| rest.split_once('.'))
    {
        index.parse::<usize>().ok()?;
        let layer = |part: &str| format!("encoder.layers.{index}.{part}");
        return match (part, suffix) {
            ("attn_rel_pos_emb", None) => renamed(layer("attn.rel_pos_emb.weight")),
            ("attn_k", Some("weight")) => Some(KeyValue {
                name: layer("attn.to_kv.weight"),
                key: true,
            }),
            ("attn_v", Some("weight")) => Some(KeyValue {
                name: layer("attn.to_kv.weight"),
                key: false,
            }),
            ("conv_pw1", Some("weight")) => Some(Unsqueezed {
                name: layer("conv.up_conv.weight"),
                dim: 2,
            }),
            ("conv_pw2", Some("weight")) => Some(Unsqueezed {
                name: layer("conv.down_conv.weight"),
                dim: 2,
            }),
            ("conv_dw", Some("weight")) => Some(Unsqueezed {
                name: layer("conv.depth_conv.conv.weight"),
                dim: 1,
            }),
            // llama.cpp folds the running statistics into weight and bias.
            ("conv_norm", Some("weight")) => renamed(layer("conv.batch_norm.folded_scale")),
            ("conv_norm", Some("bias")) => renamed(layer("conv.batch_norm.folded_shift")),
            (_, Some(_)) => {
                let base = match part {
                    "ln1" => "attn.pre_norm",
                    "ln2" => "post_norm",
                    "ffn_norm" => "ff1.pre_norm",
                    "ffn_up" => "ff1.up_proj",
                    "ffn_down" => "ff1.down_proj",
                    "ffn_norm_1" => "ff2.pre_norm",
                    "ffn_up_1" => "ff2.up_proj",
                    "ffn_down_1" => "ff2.down_proj",
                    "attn_q" => "attn.to_q",
                    "attn_out" => "attn.to_out",
                    "norm_conv" => "conv.norm",
                    "conv_pw1" => "conv.up_conv",
                    "conv_pw2" => "conv.down_conv",
                    _ => return None,
                };
                renamed(with_suffix(&layer(base)))
            }
            _ => None,
        };
    }
    if let Some((index, part)) = stem
        .strip_prefix("a.proj_blk.")
        .and_then(|rest| rest.split_once('.'))
    {
        index.parse::<usize>().ok()?;
        let base = match part {
            "self_attn_q" => "attention.attention.query",
            "self_attn_k" => "attention.attention.key",
            "self_attn_v" => "attention.attention.value",
            "self_attn_out" => "attention.output.dense",
            "self_attn_norm" => "attention.output.LayerNorm",
            "cross_attn_q" => "crossattention.attention.query",
            "cross_attn_k" => "crossattention.attention.key",
            "cross_attn_v" => "crossattention.attention.value",
            "cross_attn_out" => "crossattention.output.dense",
            "cross_attn_norm" => "crossattention.output.LayerNorm",
            "ffn_up" => "intermediate_query.dense",
            "ffn_down" => "output_query.dense",
            "ffn_norm" => "output_query.LayerNorm",
            _ => return None,
        };
        suffix?;
        return renamed(with_suffix(&format!(
            "projector.qformer.encoder.layer.{index}.{base}"
        )));
    }
    None
}

/// The safetensors name mmproj tensor `gguf_name` loads as (both halves of
/// `to_kv` name it), for load admission's pricing.
pub(crate) fn granite_speech5_mmproj_safetensors_name(gguf_name: &str) -> Option<String> {
    Some(match mmproj_tensor(gguf_name)? {
        MmprojTensor::Renamed(name)
        | MmprojTensor::Unsqueezed { name, .. }
        | MmprojTensor::KeyValue { name, .. } => name,
    })
}

/// How the builder makes one safetensors-named tensor from the mmproj.
#[derive(Debug, Clone)]
enum MmprojSource {
    Tensor(String),
    Unsqueezed { gguf: String, dim: usize },
    KeyValue { key: String, value: String },
}

/// A `VarBuilder` backend over the mmproj: each tensor, under its
/// safetensors name, is read on the host, rebuilt (K and V joined, unit
/// dimensions restored), converted to the requested dtype there and uploaded
/// when a loader asks for it, so no more than one tensor's host copies are
/// alive at a time.
struct MmprojBackend {
    gguf: GgufLoader,
    tensors: HashMap<String, MmprojSource>,
}

impl MmprojBackend {
    /// Every mmproj tensor must have a counterpart, and `to_kv` both halves,
    /// so a conversion that changes the layout fails here instead of loading
    /// a different model.
    fn new(gguf: GgufLoader) -> Result<Self> {
        let mut tensors = HashMap::new();
        let mut halves: HashMap<String, [Option<String>; 2]> = HashMap::new();
        for gguf_name in gguf.tensor_names() {
            let mapped = mmproj_tensor(&gguf_name).ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Granite Speech 5.0 mmproj tensor {gguf_name} has no counterpart in the model"
                ))
            })?;
            match mapped {
                MmprojTensor::Renamed(name) => {
                    tensors.insert(name, MmprojSource::Tensor(gguf_name));
                }
                MmprojTensor::Unsqueezed { name, dim } => {
                    tensors.insert(
                        name,
                        MmprojSource::Unsqueezed {
                            gguf: gguf_name,
                            dim,
                        },
                    );
                }
                MmprojTensor::KeyValue { name, key } => {
                    halves.entry(name).or_default()[usize::from(!key)] = Some(gguf_name);
                }
            }
        }
        for (name, halves) in halves {
            let [Some(key), Some(value)] = halves else {
                return Err(Error::ModelLoadError(format!(
                    "Granite Speech 5.0 mmproj has only one of the K and V halves of {name}"
                )));
            };
            tensors.insert(name, MmprojSource::KeyValue { key, value });
        }
        Ok(Self { gguf, tensors })
    }

    /// Tensor `name` on the host, in its stored dtype.
    fn read(&self, name: &str) -> Result<Tensor> {
        let source = self.tensors.get(name).ok_or_else(|| {
            Error::ModelLoadError(format!(
                "Granite Speech 5.0 mmproj has no tensor for {name}"
            ))
        })?;
        match source {
            MmprojSource::Tensor(gguf) => host_tensor(&self.gguf, gguf),
            MmprojSource::Unsqueezed { gguf, dim } => host_tensor(&self.gguf, gguf)?
                .unsqueeze(*dim)
                .map_err(Error::from),
            MmprojSource::KeyValue { key, value } => Tensor::cat(
                &[
                    host_tensor(&self.gguf, key)?,
                    host_tensor(&self.gguf, value)?,
                ],
                0,
            )
            .map_err(Error::from),
        }
    }
}

impl candle_nn::var_builder::SimpleBackend for MmprojBackend {
    fn get(
        &self,
        shape: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!(
                "Granite Speech 5.0 mmproj tensor {name} is {:?}, expected {shape:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.read(name)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .to_dtype(dtype)?
            .to_device(device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// A builder over the mmproj's tensors under their safetensors names
/// (`encoder.*`, `projector.*`) at `dtype` on `device`.
pub(super) fn mmproj_var_builder(
    gguf: GgufLoader,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    Ok(VarBuilder::from_backend(
        Box::new(MmprojBackend::new(gguf)?),
        dtype,
        device.clone(),
    ))
}

/// A float mmproj tensor on the host in its stored dtype.
fn host_tensor(gguf: &GgufLoader, name: &str) -> Result<Tensor> {
    let qtensor = gguf.load_qtensor(name, &Device::Cpu)?;
    match qtensor.dtype() {
        GgmlDType::F16 => qtensor.dequantize_f16(&Device::Cpu),
        GgmlDType::F32 | GgmlDType::BF16 => qtensor.dequantize(&Device::Cpu),
        other => {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 mmproj tensor {name} is {other:?}; the loader reads float mmproj tensors"
            )))
        }
    }
    .map_err(Error::from)
}

/// One language-model tensor joined into a projection: its GGUF name and, for
/// Q and K, the heads whose rows llama.cpp reordered for interleaved RoPE.
pub(super) struct QuantizedPart<'a> {
    pub(super) name: &'a str,
    pub(super) rope_heads: Option<usize>,
}

/// The parts' rows joined, in order, into one quantized projection on
/// `device`, with Q and K rows put back in half-split RoPE order.
pub(super) fn joined_qmatmul(
    gguf: &GgufLoader,
    parts: &[QuantizedPart<'_>],
    device: &Device,
) -> Result<QMatMul> {
    QMatMul::from_qtensor(joined_qtensor(gguf, parts, device)?).map_err(Error::from)
}

/// [`joined_qmatmul`]'s tensor. Each part is read on the host and the joined
/// rows uploaded once.
pub(super) fn joined_qtensor(
    gguf: &GgufLoader,
    parts: &[QuantizedPart<'_>],
    device: &Device,
) -> Result<QTensor> {
    let mut joined = Vec::new();
    let mut rows = 0;
    let mut layout: Option<(GgmlDType, usize)> = None;
    for part in parts {
        let tensor = gguf.load_qtensor(part.name, &Device::Cpu)?;
        let (part_rows, columns) = tensor.shape().dims2()?;
        match layout {
            None => layout = Some((tensor.dtype(), columns)),
            Some(expected) if expected == (tensor.dtype(), columns) => {}
            Some((dtype, width)) => {
                return Err(Error::ModelLoadError(format!(
                    "Granite Speech 5.0 GGUF tensor {} is {:?} x {columns}, unlike the {dtype:?} x {width} it joins",
                    part.name,
                    tensor.dtype()
                )))
            }
        }
        let data = tensor.data()?;
        let order = match part.rope_heads {
            Some(heads) => half_split_row_order(part_rows, heads).ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Granite Speech 5.0 GGUF tensor {} has {part_rows} rows, which do not split into {heads} heads of even width",
                    part.name
                ))
            })?,
            None => (0..part_rows).collect(),
        };
        append_rows(&mut joined, &data, part_rows, &order)?;
        rows += part_rows;
    }
    let (dtype, columns) = layout.ok_or_else(|| {
        Error::ModelLoadError("Granite Speech 5.0 GGUF projection has no parts".into())
    })?;
    let storage = QStorage::from_data(Cow::Owned(joined), device, dtype)?;
    QTensor::new(storage, (rows, columns)).map_err(Error::from)
}

/// The GGUF row that holds each half-split row, head by head. llama.cpp's
/// conversion stores row `i` of a head's first half at `2i` and row `i` of
/// its second half at `2i + 1`.
fn half_split_row_order(rows: usize, heads: usize) -> Option<Vec<usize>> {
    if heads == 0 || !rows.is_multiple_of(heads) {
        return None;
    }
    let head_rows = rows / heads;
    if !head_rows.is_multiple_of(2) {
        return None;
    }
    let half = head_rows / 2;
    Some(
        (0..heads)
            .flat_map(|head| {
                (0..2).flat_map(move |second| {
                    (0..half).map(move |index| head * head_rows + 2 * index + second)
                })
            })
            .collect(),
    )
}

/// Appends `data`'s rows (`rows` of equal byte length) in `order`.
fn append_rows(out: &mut Vec<u8>, data: &[u8], rows: usize, order: &[usize]) -> Result<()> {
    if rows == 0 || !data.len().is_multiple_of(rows) {
        return Err(Error::ModelLoadError(format!(
            "a GGUF tensor of {} bytes does not hold {rows} equal rows",
            data.len()
        )));
    }
    let row_bytes = data.len() / rows;
    out.reserve(data.len());
    for &row in order {
        out.extend_from_slice(&data[row * row_bytes..(row + 1) * row_bytes]);
    }
    Ok(())
}

/// Opens a GGUF file for reading on the host.
pub(super) fn open_gguf(path: &Path) -> Result<GgufLoader> {
    GgufLoader::from_path_with_backend(path, BackendKind::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_split_order_undoes_llamas_interleave() {
        // llama.cpp's `permute` on one head of 6 rows: [a0 a1 a2 | b0 b1 b2]
        // becomes [a0 b0 a1 b1 a2 b2].
        let interleaved = ["a0", "b0", "a1", "b1", "a2", "b2"];
        let order = half_split_row_order(6, 1).unwrap();
        let restored = order
            .iter()
            .map(|row| interleaved[*row])
            .collect::<Vec<_>>();
        assert_eq!(restored, ["a0", "a1", "a2", "b0", "b1", "b2"]);
        // Heads stay in place.
        assert_eq!(
            half_split_row_order(8, 2).unwrap(),
            [0, 2, 1, 3, 4, 6, 5, 7]
        );
        assert!(half_split_row_order(6, 4).is_none());
        assert!(half_split_row_order(6, 2).is_none());
    }

    #[test]
    fn joined_rows_follow_the_order() {
        let data = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let mut out = Vec::new();
        append_rows(&mut out, &data, 4, &[2, 0, 3, 1]).unwrap();
        assert_eq!(out, [2, 2, 0, 0, 3, 3, 1, 1]);
        assert!(append_rows(&mut out, &data[..7], 4, &[0]).is_err());
    }

    #[test]
    fn every_mmproj_tensor_maps_to_a_model_tensor() {
        use MmprojTensor::*;
        for (gguf, expected) in [
            (
                "a.input_projection.weight",
                Renamed("encoder.input_linear.weight".into()),
            ),
            (
                "a.enc_ctc_out_mid.bias",
                Renamed("encoder.out_mid.bias".into()),
            ),
            (
                "a.blk.3.ln1.weight",
                Renamed("encoder.layers.3.attn.pre_norm.weight".into()),
            ),
            (
                "a.blk.3.ln2.bias",
                Renamed("encoder.layers.3.post_norm.bias".into()),
            ),
            (
                "a.blk.3.ffn_up_1.weight",
                Renamed("encoder.layers.3.ff2.up_proj.weight".into()),
            ),
            (
                "a.blk.3.ffn_norm.bias",
                Renamed("encoder.layers.3.ff1.pre_norm.bias".into()),
            ),
            (
                "a.blk.3.attn_rel_pos_emb",
                Renamed("encoder.layers.3.attn.rel_pos_emb.weight".into()),
            ),
            (
                "a.blk.3.attn_v.weight",
                KeyValue {
                    name: "encoder.layers.3.attn.to_kv.weight".into(),
                    key: false,
                },
            ),
            (
                "a.blk.3.conv_dw.weight",
                Unsqueezed {
                    name: "encoder.layers.3.conv.depth_conv.conv.weight".into(),
                    dim: 1,
                },
            ),
            (
                "a.blk.3.conv_pw1.bias",
                Renamed("encoder.layers.3.conv.up_conv.bias".into()),
            ),
            (
                "a.blk.3.conv_norm.weight",
                Renamed("encoder.layers.3.conv.batch_norm.folded_scale".into()),
            ),
            (
                "a.blk.3.norm_conv.weight",
                Renamed("encoder.layers.3.conv.norm.weight".into()),
            ),
            ("a.proj_query", Renamed("projector.query".into())),
            (
                "a.proj_blk.1.cross_attn_norm.bias",
                Renamed(
                    "projector.qformer.encoder.layer.1.crossattention.output.LayerNorm.bias".into(),
                ),
            ),
            (
                "a.proj_blk.1.ffn_down.weight",
                Renamed("projector.qformer.encoder.layer.1.output_query.dense.weight".into()),
            ),
        ] {
            assert_eq!(mmproj_tensor(gguf), Some(expected), "{gguf}");
        }
        for unknown in [
            "a.blk.3.attn_k.bias",
            "a.blk.x.ln1.weight",
            "v.blk.0.ln1.weight",
            "a.proj_blk.1.unknown.weight",
        ] {
            assert_eq!(mmproj_tensor(unknown), None, "{unknown}");
        }
    }
}

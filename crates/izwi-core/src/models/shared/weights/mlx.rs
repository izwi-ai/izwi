//! Compatibility helpers for MLX-exported safetensors checkpoints.
//!
//! MLX quantized checkpoints store affine-packed weights as:
//! - `weight`  (U8/U32 packed values)
//! - `scales`  (per-group scale)
//! - `biases`  (per-group bias)
//!
//! This module dequantizes those tensors into dense weights so existing Candle
//! layers can be reused without model-specific branching.

use candle_core::{DType, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, Linear, VarBuilder};

use crate::error::{Error, Result};

fn tensor2_f32(t: Tensor) -> Result<Vec<Vec<f32>>> {
    let t = t.to_dtype(DType::F32)?;
    t.to_vec2::<f32>().map_err(Error::from)
}

fn transpose_2d(mut rows: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    if rows.is_empty() {
        return Err(Error::ModelLoadError(
            "Empty 2D tensor for quantization parameters".to_string(),
        ));
    }
    let cols = rows
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty 2D tensor row".to_string()))?;
    if cols == 0 {
        return Err(Error::ModelLoadError(
            "Empty 2D tensor for quantization parameters".to_string(),
        ));
    }
    if rows.iter().any(|r| r.len() != cols) {
        return Err(Error::ModelLoadError(
            "Ragged 2D tensor for quantization parameters".to_string(),
        ));
    }
    let mut out = vec![vec![0.0f32; rows.len()]; cols];
    for (i, row) in rows.iter_mut().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[j][i] = *v;
        }
    }
    Ok(out)
}

fn normalize_affine_params(
    scales: Vec<Vec<f32>>,
    biases: Option<Vec<Vec<f32>>>,
    out_dim: usize,
    in_dim: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, usize)> {
    if scales.is_empty() {
        return Err(Error::ModelLoadError(
            "Empty quantization scales".to_string(),
        ));
    }

    let (scales, biases, groups) = if scales.len() == out_dim {
        let groups = scales
            .first()
            .map(|r| r.len())
            .ok_or_else(|| Error::ModelLoadError("Empty quantization scales".to_string()))?;
        (scales, biases, groups)
    } else if scales.first().map(|r| r.len()) == Some(out_dim) {
        let groups = scales.len();
        let scales = transpose_2d(scales)?;
        let biases = match biases {
            Some(biases) => Some(transpose_2d(biases)?),
            None => None,
        };
        (scales, biases, groups)
    } else {
        return Err(Error::ModelLoadError(format!(
            "Quantization scales shape mismatch: expected out_dim={out_dim}, got rows={} cols={}",
            scales.len(),
            scales.first().map(|r| r.len()).unwrap_or(0)
        )));
    };

    if groups == 0 || !in_dim.is_multiple_of(groups) {
        return Err(Error::ModelLoadError(format!(
            "Invalid quantization groups: in_dim={in_dim}, groups={groups}"
        )));
    }
    if scales.iter().any(|r| r.len() != groups) {
        return Err(Error::ModelLoadError(
            "Ragged quantization scales".to_string(),
        ));
    }

    let biases = match biases {
        Some(biases) => {
            if biases.len() != out_dim || biases.iter().any(|r| r.len() != groups) {
                return Err(Error::ModelLoadError(format!(
                    "Quantization biases shape mismatch: expected ({out_dim},{groups})"
                )));
            }
            biases
        }
        None => vec![vec![0.0f32; groups]; out_dim],
    };

    Ok((scales, biases, groups))
}

fn quant_bits_from_packing(
    expected_in_dim: usize,
    packed_in_dim: usize,
    word_bits: usize,
) -> Result<usize> {
    if packed_in_dim == 0 || expected_in_dim == 0 {
        return Err(Error::ModelLoadError(
            "Invalid packed shape for quantized weight".to_string(),
        ));
    }
    if !expected_in_dim.is_multiple_of(packed_in_dim) {
        return Err(Error::ModelLoadError(format!(
            "Quantized weight has incompatible input dim: expected_in={expected_in_dim}, packed_in={packed_in_dim}"
        )));
    }
    let pack_factor = expected_in_dim / packed_in_dim;
    if !word_bits.is_multiple_of(pack_factor) {
        return Err(Error::ModelLoadError(format!(
            "Invalid quant packing factor: word_bits={word_bits}, pack_factor={pack_factor}"
        )));
    }
    let bits = word_bits / pack_factor;
    if bits == 0 || bits > 8 {
        return Err(Error::ModelLoadError(format!(
            "Unsupported quantization bit-width inferred from packing: {bits}"
        )));
    }
    Ok(bits)
}

fn dequantize_affine_u32(
    packed: Vec<Vec<u32>>,
    scales: Vec<Vec<f32>>,
    biases: Option<Vec<Vec<f32>>>,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if packed.len() != out_dim {
        return Err(Error::ModelLoadError(format!(
            "Quantized weight rows mismatch: expected out_dim={out_dim}, packed_rows={}",
            packed.len()
        )));
    }
    let packed_in = packed
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty packed quantized weight".to_string()))?;
    let bits = quant_bits_from_packing(in_dim, packed_in, 32)?;
    let pack_factor = in_dim / packed_in;
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };

    let (scales, biases, groups) = normalize_affine_params(scales, biases, out_dim, in_dim)?;
    let group_size = in_dim / groups;

    let mut out = vec![0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for pcol in 0..packed_in {
            let word = packed[row][pcol];
            let base_col = pcol * pack_factor;
            for pi in 0..pack_factor {
                let col = base_col + pi;
                if col >= in_dim {
                    break;
                }
                let q = ((word >> (pi * bits)) & mask) as f32;
                let g = col / group_size;
                out[row * in_dim + col] = q * scales[row][g] + biases[row][g];
            }
        }
    }
    Ok(out)
}

fn dequantize_affine_u8(
    packed: Vec<Vec<u8>>,
    scales: Vec<Vec<f32>>,
    biases: Option<Vec<Vec<f32>>>,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if packed.len() != out_dim {
        return Err(Error::ModelLoadError(format!(
            "Quantized weight rows mismatch: expected out_dim={out_dim}, packed_rows={}",
            packed.len()
        )));
    }
    let packed_in = packed
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty packed quantized weight".to_string()))?;
    let bits = quant_bits_from_packing(in_dim, packed_in, 8)?;
    let pack_factor = in_dim / packed_in;
    let mask = if bits == 8 { 0xFF } else { (1u32 << bits) - 1 };

    let (scales, biases, groups) = normalize_affine_params(scales, biases, out_dim, in_dim)?;
    let group_size = in_dim / groups;

    let mut out = vec![0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for pcol in 0..packed_in {
            let word = packed[row][pcol] as u32;
            let base_col = pcol * pack_factor;
            for pi in 0..pack_factor {
                let col = base_col + pi;
                if col >= in_dim {
                    break;
                }
                let q = ((word >> (pi * bits)) & mask) as f32;
                let g = col / group_size;
                out[row * in_dim + col] = q * scales[row][g] + biases[row][g];
            }
        }
    }
    Ok(out)
}

pub fn has_mlx_affine_quantized_weight(vb: &VarBuilder) -> bool {
    vb.contains_tensor("weight") && vb.contains_tensor("scales")
}

pub fn load_weight(vb: &VarBuilder, out_dim: usize, in_dim: usize) -> Result<Tensor> {
    if !has_mlx_affine_quantized_weight(vb) {
        return vb.get((out_dim, in_dim), "weight").map_err(Error::from);
    }

    let scales = tensor2_f32(vb.get_unchecked_dtype("scales", DType::F32)?)?;
    let biases = if vb.contains_tensor("biases") {
        Some(tensor2_f32(vb.get_unchecked_dtype("biases", DType::F32)?)?)
    } else {
        None
    };

    let weight = if let Ok(w_u32) = vb.get_unchecked_dtype("weight", DType::U32) {
        let packed = w_u32.to_vec2::<u32>()?;
        dequantize_affine_u32(packed, scales, biases, out_dim, in_dim)?
    } else if let Ok(w_u8) = vb.get_unchecked_dtype("weight", DType::U8) {
        let packed = w_u8.to_vec2::<u8>()?;
        dequantize_affine_u8(packed, scales, biases, out_dim, in_dim)?
    } else {
        return Err(Error::ModelLoadError(
            "MLX quantized tensor has unsupported packed dtype for `weight` (expected U32 or U8)"
                .to_string(),
        ));
    };

    let t = Tensor::from_vec(weight, (out_dim, in_dim), vb.device())?;
    t.to_dtype(vb.dtype()).map_err(Error::from)
}

pub fn load_linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if has_mlx_affine_quantized_weight(&vb) {
        let ws = load_weight(&vb, out_dim, in_dim)?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        Ok(Linear::new(ws, bias))
    } else {
        candle_nn::linear(in_dim, out_dim, vb).map_err(Error::from)
    }
}

pub fn load_linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if has_mlx_affine_quantized_weight(&vb) {
        let ws = load_weight(&vb, out_dim, in_dim)?;
        Ok(Linear::new(ws, None))
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb).map_err(Error::from)
    }
}

pub fn load_embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    if has_mlx_affine_quantized_weight(&vb) {
        let ws = load_weight(&vb, vocab_size, hidden_size)?;
        Ok(Embedding::new(ws, hidden_size))
    } else {
        candle_nn::embedding(vocab_size, hidden_size, vb).map_err(Error::from)
    }
}

pub fn load_conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let mut ws = vb.get_unchecked_dtype("weight", vb.dtype())?;

    let dims = ws.dims4()?;
    let expected_oihw = (out_channels, in_channels, kernel_size, kernel_size);
    let expected_ohwi = (out_channels, kernel_size, kernel_size, in_channels);

    if dims == expected_ohwi {
        ws = ws.permute((0, 3, 1, 2))?;
    } else if dims != expected_oihw {
        return Err(Error::ModelLoadError(format!(
            "Conv2d weight shape mismatch: got={dims:?}, expected OIHW={expected_oihw:?} or OHWI={expected_ohwi:?}"
        )));
    }

    let bias = if vb.contains_tensor("bias") {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv2d::new(ws, bias, cfg))
}

pub fn load_conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    if cfg.groups == 0 || !in_channels.is_multiple_of(cfg.groups) {
        return Err(Error::ModelLoadError(format!(
            "Invalid Conv1d groups for shape inference: in_channels={in_channels}, groups={}",
            cfg.groups
        )));
    }

    let in_per_group = in_channels / cfg.groups;
    let expected_oik = (out_channels, in_per_group, kernel_size);
    let expected_oki = (out_channels, kernel_size, in_per_group);

    let mut ws = vb.get_unchecked_dtype("weight", vb.dtype())?;
    if let Ok(dims) = ws.dims3() {
        if dims == expected_oki {
            ws = ws.permute((0, 2, 1))?;
        } else if dims != expected_oik {
            return Err(Error::ModelLoadError(format!(
                "Conv1d weight shape mismatch: got={dims:?}, expected OIK={expected_oik:?} or OKI={expected_oki:?}"
            )));
        }
    } else if let Ok(dims2) = ws.dims2() {
        if kernel_size != 1 || dims2 != (out_channels, in_per_group) {
            return Err(Error::ModelLoadError(format!(
                "Conv1d 2D weight shape mismatch: got={dims2:?}, expected ({out_channels},{in_per_group}) for kernel_size=1"
            )));
        }
        ws = ws.unsqueeze(2)?;
    } else {
        return Err(Error::ModelLoadError(
            "Conv1d weight must be a 2D/3D tensor".to_string(),
        ));
    }

    Ok(Conv1d::new(ws, None, cfg))
}

pub fn load_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    if cfg.groups == 0 || !in_channels.is_multiple_of(cfg.groups) {
        return Err(Error::ModelLoadError(format!(
            "Invalid Conv1d groups for shape inference: in_channels={in_channels}, groups={}",
            cfg.groups
        )));
    }

    let in_per_group = in_channels / cfg.groups;
    let expected_oik = (out_channels, in_per_group, kernel_size);
    let expected_oki = (out_channels, kernel_size, in_per_group);

    let mut ws = vb.get_unchecked_dtype("weight", vb.dtype())?;
    if let Ok(dims) = ws.dims3() {
        if dims == expected_oki {
            ws = ws.permute((0, 2, 1))?;
        } else if dims != expected_oik {
            return Err(Error::ModelLoadError(format!(
                "Conv1d weight shape mismatch: got={dims:?}, expected OIK={expected_oik:?} or OKI={expected_oki:?}"
            )));
        }
    } else if let Ok(dims2) = ws.dims2() {
        if kernel_size != 1 || dims2 != (out_channels, in_per_group) {
            return Err(Error::ModelLoadError(format!(
                "Conv1d 2D weight shape mismatch: got={dims2:?}, expected ({out_channels},{in_per_group}) for kernel_size=1"
            )));
        }
        ws = ws.unsqueeze(2)?;
    } else {
        return Err(Error::ModelLoadError(
            "Conv1d weight must be a 2D/3D tensor".to_string(),
        ));
    }

    let bias = if vb.contains_tensor("bias") {
        Some(vb.get_unchecked_dtype("bias", vb.dtype())?)
    } else {
        None
    };

    Ok(Conv1d::new(ws, bias, cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_u32_4bit_affine() {
        let packed = vec![vec![0x7654_3210]];
        let scales = vec![vec![0.1, 0.2]];
        let biases = vec![vec![-1.0, 0.5]];
        let out = dequantize_affine_u32(packed, scales, Some(biases), 1, 8).unwrap();

        let expected = [
            -1.0 + 0.0 * 0.1,
            -1.0 + 1.0 * 0.1,
            -1.0 + 2.0 * 0.1,
            -1.0 + 3.0 * 0.1,
            0.5 + 4.0 * 0.2,
            0.5 + 5.0 * 0.2,
            0.5 + 6.0 * 0.2,
            0.5 + 7.0 * 0.2,
        ];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dequantize_u8_8bit_affine() {
        let packed = vec![vec![1u8, 2, 3, 4]];
        let scales = vec![vec![0.5, 2.0]];
        let biases = vec![vec![0.0, -1.0]];
        let out = dequantize_affine_u8(packed, scales, Some(biases), 1, 4).unwrap();

        let expected = [0.5, 1.0, 5.0, 7.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dequantize_u32_affine_no_biases() {
        let packed = vec![vec![0x7654_3210]];
        let scales = vec![vec![0.1, 0.2]];
        let out = dequantize_affine_u32(packed, scales, None, 1, 8).unwrap();

        let expected = [
            0.0 * 0.1,
            1.0 * 0.1,
            2.0 * 0.1,
            3.0 * 0.1,
            4.0 * 0.2,
            5.0 * 0.2,
            6.0 * 0.2,
            7.0 * 0.2,
        ];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dequantize_u8_affine_no_biases() {
        let packed = vec![vec![1u8, 2, 3, 4]];
        let scales = vec![vec![0.5, 2.0]];
        let out = dequantize_affine_u8(packed, scales, None, 1, 4).unwrap();

        let expected = [0.5, 1.0, 6.0, 8.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

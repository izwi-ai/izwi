//! Compatibility helpers for MLX-exported safetensors checkpoints.
//!
//! MLX quantized checkpoints store affine-packed weights as:
//! - `weight`  (U8/U32 packed values)
//! - `scales`  (per-group scale)
//! - `biases`  (per-group bias)
//!
//! This module dequantizes those tensors into dense weights so existing Candle
//! layers can be reused without model-specific branching.

use candle_core::{DType, Device, Tensor, WithDType};
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

fn dequantize_affine_u32<T: WithDType>(
    packed: Vec<Vec<u32>>,
    scales: Vec<Vec<f32>>,
    biases: Option<Vec<Vec<f32>>>,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<T>> {
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

    let mut out = vec![T::zero(); out_dim * in_dim];
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
                out[row * in_dim + col] =
                    T::from_f64(f64::from(q * scales[row][g] + biases[row][g]));
            }
        }
    }
    Ok(out)
}

fn dequantize_affine_u8<T: WithDType>(
    packed: Vec<Vec<u8>>,
    scales: Vec<Vec<f32>>,
    biases: Option<Vec<Vec<f32>>>,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<T>> {
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

    let mut out = vec![T::zero(); out_dim * in_dim];
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
                out[row * in_dim + col] =
                    T::from_f64(f64::from(q * scales[row][g] + biases[row][g]));
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

    let packed = if let Ok(w_u32) = vb.get_unchecked_dtype("weight", DType::U32) {
        PackedWords::U32(w_u32.to_vec2::<u32>()?)
    } else if let Ok(w_u8) = vb.get_unchecked_dtype("weight", DType::U8) {
        PackedWords::U8(w_u8.to_vec2::<u8>()?)
    } else {
        return Err(Error::ModelLoadError(
            "MLX quantized tensor has unsupported packed dtype for `weight` (expected U32 or U8)"
                .to_string(),
        ));
    };

    // Dequantize on the host straight into the requested dtype and upload the
    // result once, at its exact size. An F32 buffer uploaded and cast on the
    // device kept two F32 copies of the largest weight in flight (the 4-bit
    // forced aligner's embedding: 0.58 GiB on the host, 0.58 GiB uploaded)
    // and rounded the cast up to a power of two.
    let (shape, device) = ((out_dim, in_dim), vb.device());
    match vb.dtype() {
        DType::F16 => packed.dequantize::<half::f16>(scales, biases, shape, device),
        DType::BF16 => packed.dequantize::<half::bf16>(scales, biases, shape, device),
        DType::F32 => packed.dequantize::<f32>(scales, biases, shape, device),
        dtype => packed
            .dequantize::<f32>(scales, biases, shape, device)?
            .to_dtype(dtype)
            .map_err(Error::from),
    }
}

/// The packed words of an MLX affine-quantized weight.
enum PackedWords {
    U32(Vec<Vec<u32>>),
    U8(Vec<Vec<u8>>),
}

impl PackedWords {
    /// The dense `shape` weight at `T`, on `device`.
    fn dequantize<T: WithDType>(
        self,
        scales: Vec<Vec<f32>>,
        biases: Option<Vec<Vec<f32>>>,
        (out_dim, in_dim): (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let values = match self {
            Self::U32(packed) => {
                dequantize_affine_u32::<T>(packed, scales, biases, out_dim, in_dim)?
            }
            Self::U8(packed) => dequantize_affine_u8::<T>(packed, scales, biases, out_dim, in_dim)?,
        };
        Tensor::from_vec(values, (out_dim, in_dim), device).map_err(Error::from)
    }
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
    // Candle 0.11's Metal and non-cudnn CUDA im2col conv2d copy a strided
    // kernel to a contiguous buffer and then multiply the original one. Both
    // the permute above and a Fortran-ordered `.pth` tensor are strided.
    let ws = if ws.is_contiguous() {
        ws
    } else {
        let contiguous = ws.contiguous()?;
        // Free the strided original now: until a synchronization Candle
        // pools it and hands it to the next allocation it fits, so the small
        // bias read next would keep the whole kernel's buffer resident.
        drop(ws);
        contiguous.device().synchronize()?;
        contiguous
    };

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
    // Candle 0.11's CPU, Metal and non-cudnn CUDA im2col conv1d copy a strided
    // kernel to a contiguous buffer and then multiply the original. Both the
    // permute above and a Fortran-ordered `.pth` tensor are strided.
    let ws = ws.contiguous()?;

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
    // Candle 0.11's CPU, Metal and non-cudnn CUDA im2col conv1d copy a strided
    // kernel to a contiguous buffer and then multiply the original. Both the
    // permute above and a Fortran-ordered `.pth` tensor are strided.
    let ws = ws.contiguous()?;

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
        let out = dequantize_affine_u32::<f32>(packed, scales, Some(biases), 1, 8).unwrap();

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
        let out = dequantize_affine_u8::<f32>(packed, scales, Some(biases), 1, 4).unwrap();

        let expected = [0.5, 1.0, 5.0, 7.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dequantize_u32_affine_no_biases() {
        let packed = vec![vec![0x7654_3210]];
        let scales = vec![vec![0.1, 0.2]];
        let out = dequantize_affine_u32::<f32>(packed, scales, None, 1, 8).unwrap();

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
        let out = dequantize_affine_u8::<f32>(packed, scales, None, 1, 4).unwrap();

        let expected = [0.5, 1.0, 6.0, 8.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    /// The host dequantization into the requested dtype matches the F32
    /// dequantization cast to it, bit for bit, at the exact shape.
    #[test]
    fn load_weight_dequantizes_into_the_requested_dtype() {
        let device = Device::Cpu;
        let (out_dim, in_dim) = (3, 16);
        let words: Vec<u32> = (0..out_dim * in_dim / 8)
            .map(|i| 0x9E37_79B9u32.wrapping_mul(i as u32 + 1))
            .collect();
        let scales: Vec<f32> = (0..out_dim * 2).map(|i| 0.013 * (i as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..out_dim * 2).map(|i| -0.07 * i as f32).collect();
        let weights = std::collections::HashMap::from([
            (
                "weight".to_string(),
                Tensor::from_vec(words, (out_dim, in_dim / 8), &device).unwrap(),
            ),
            (
                "scales".to_string(),
                Tensor::from_vec(scales, (out_dim, 2), &device).unwrap(),
            ),
            (
                "biases".to_string(),
                Tensor::from_vec(biases, (out_dim, 2), &device).unwrap(),
            ),
        ]);
        let vb = |dtype| VarBuilder::from_tensors(weights.clone(), dtype, &device);
        let reference = load_weight(&vb(DType::F32), out_dim, in_dim).unwrap();
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let weight = load_weight(&vb(dtype), out_dim, in_dim).unwrap();
            assert_eq!(
                (weight.dtype(), weight.dims2().unwrap()),
                (dtype, (out_dim, in_dim))
            );
            let expected = reference.to_dtype(dtype).unwrap();
            let bits = |t: &Tensor| {
                t.flatten_all()
                    .unwrap()
                    .to_dtype(DType::F64)
                    .unwrap()
                    .to_vec1::<f64>()
                    .unwrap()
            };
            assert_eq!(bits(&weight), bits(&expected), "{dtype:?}");
        }
    }

    /// MLX OHWI / OKI conv weights for the loaders, with a bias.
    fn mlx_conv_weights(
        out_channels: usize,
        kernel: &[usize],
        in_channels: usize,
    ) -> std::collections::HashMap<String, Tensor> {
        use crate::kernels::numeric_check::xorshift_tensor;

        let mut shape = vec![out_channels];
        shape.extend_from_slice(kernel);
        shape.push(in_channels);
        let device = candle_core::Device::Cpu;
        std::collections::HashMap::from([
            (
                "weight".to_string(),
                xorshift_tensor(&shape, 9, 0.05, &device).expect("weight"),
            ),
            (
                "bias".to_string(),
                xorshift_tensor(&[out_channels], 11, 0.5, &device).expect("bias"),
            ),
        ])
    }

    #[test]
    fn mlx_conv_loaders_return_contiguous_oihw_kernels() {
        let device = candle_core::Device::Cpu;
        let vb = |weights| VarBuilder::from_tensors(weights, DType::F32, &device);
        let conv2d = load_conv2d(
            8,
            6,
            3,
            Conv2dConfig::default(),
            vb(mlx_conv_weights(6, &[3, 3], 8)),
        )
        .expect("conv2d");
        assert_eq!(conv2d.weight().dims(), &[6, 8, 3, 3]);
        assert!(conv2d.weight().is_contiguous());

        let weights = mlx_conv_weights(6, &[3], 8);
        let conv1d =
            load_conv1d(8, 6, 3, Conv1dConfig::default(), vb(weights.clone())).expect("conv1d");
        let no_bias = load_conv1d_no_bias(8, 6, 3, Conv1dConfig::default(), vb(weights))
            .expect("conv1d without bias");
        for weight in [conv1d.weight(), no_bias.weight()] {
            assert_eq!(weight.dims(), &[6, 8, 3]);
            assert!(weight.is_contiguous());
        }
    }

    /// Candle 0.11's Metal im2col conv multiplies the caller's strided kernel
    /// instead of its own contiguous copy, so a permuted MLX kernel was read in
    /// the wrong layout (relative error ~1 in the 4-bit aligner's audio CNN);
    /// its CPU conv1d ignores kernel strides too. The reference is a CPU conv
    /// over an explicitly contiguous OIHW/OIK kernel; Metal legs run when present.
    #[test]
    fn mlx_conv2d_ohwi_kernel_matches_cpu() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};
        use candle_core::{Device, Module};

        let cpu = Device::Cpu;
        // The Metal leg holds the shared device's test lease throughout.
        #[cfg(feature = "metal")]
        let metal =
            crate::backends::metal_test_device_or_skip("mlx_conv2d_ohwi_kernel_matches_cpu");
        #[cfg(not(feature = "metal"))]
        let metal: Option<(Device, ())> = None;
        let devices = std::iter::once(("cpu", cpu.clone()))
            .chain(metal.as_ref().map(|(metal, _)| ("metal", metal.clone())))
            .collect::<Vec<_>>();

        let conv2d_weights = mlx_conv_weights(480, &[3, 3], 480);
        let conv2d_input = xorshift_tensor(&[2, 480, 32, 25], 7, 1.0, &cpu).unwrap();
        let conv2d_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let conv2d_on = |device: &Device| {
            let vb = VarBuilder::from_tensors(conv2d_weights.clone(), DType::F32, device);
            let conv = load_conv2d(480, 480, 3, conv2d_cfg, vb).expect("conv2d");
            host_f32(
                &conv
                    .forward(&conv2d_input.to_device(device).unwrap())
                    .unwrap(),
            )
            .unwrap()
        };
        let oihw = conv2d_weights["weight"].permute((0, 3, 1, 2)).unwrap();
        let conv2d_reference = Conv2d::new(
            oihw.contiguous().unwrap(),
            Some(conv2d_weights["bias"].clone()),
            conv2d_cfg,
        );
        let conv2d_want = host_f32(&conv2d_reference.forward(&conv2d_input).unwrap()).unwrap();

        let conv1d_weights = mlx_conv_weights(256, &[3], 128);
        let conv1d_input = xorshift_tensor(&[2, 128, 50], 5, 1.0, &cpu).unwrap();
        let conv1d_cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1d_on = |device: &Device| {
            let vb = VarBuilder::from_tensors(conv1d_weights.clone(), DType::F32, device);
            let conv = load_conv1d(128, 256, 3, conv1d_cfg, vb).expect("conv1d");
            host_f32(
                &conv
                    .forward(&conv1d_input.to_device(device).unwrap())
                    .unwrap(),
            )
            .unwrap()
        };
        let oik = conv1d_weights["weight"].permute((0, 2, 1)).unwrap();
        let conv1d_reference = Conv1d::new(
            oik.contiguous().unwrap(),
            Some(conv1d_weights["bias"].clone()),
            conv1d_cfg,
        );
        let conv1d_want = host_f32(&conv1d_reference.forward(&conv1d_input).unwrap()).unwrap();

        let mut failures = Vec::new();
        for (device_name, device) in &devices {
            for (conv, want, got) in [
                ("conv2d", &conv2d_want, conv2d_on(device)),
                ("conv1d", &conv1d_want, conv1d_on(device)),
            ] {
                let err = max_abs_diff(&got, want);
                eprintln!("{conv} {device_name}: max|d|={err:e} vs contiguous-kernel reference");
                if err >= 1e-3 {
                    failures.push(format!("{conv} {device_name}: max|d|={err:e}"));
                }
            }
        }
        assert!(failures.is_empty(), "MLX conv loaders: {failures:?}");
    }

    /// Pins Candle 0.11's strided-kernel bug and izwi's guard against it.
    /// Candle's CPU im2col conv1d copies a strided kernel to a contiguous
    /// buffer and then multiplies the original storage as if it were row-major
    /// (Metal and non-cudnn CUDA do the same for conv1d and conv2d). If the
    /// first assertion fails, a Candle upgrade changed that: re-audit before
    /// relying on it. Every izwi conv loader must hand Candle contiguous
    /// kernels, from MLX OKI/OHWI layouts and from Fortran-ordered `.pth`.
    #[test]
    fn candle_conv_strided_kernel_canary() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};
        use crate::models::shared::weights::pytorch::test_support::write_f32_state_dict;
        use candle_core::{Device, Module};

        let cpu = Device::Cpu;
        let input = xorshift_tensor(&[2, 8, 20], 5, 1.0, &cpu).unwrap();
        let oki = xorshift_tensor(&[6, 3, 8], 9, 0.05, &cpu).unwrap();
        let strided = oki.permute((0, 2, 1)).unwrap();
        assert!(!strided.is_contiguous());
        let oik = strided.contiguous().unwrap();
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let want = host_f32(&Conv1d::new(oik.clone(), None, cfg).forward(&input).unwrap()).unwrap();
        let buggy = host_f32(&Conv1d::new(strided, None, cfg).forward(&input).unwrap()).unwrap();
        assert!(
            max_abs_diff(&buggy, &want) > 1e-2,
            "Candle CPU conv1d now honours strided kernels; re-audit report 09"
        );

        let dir = std::env::temp_dir().join(format!("izwi-conv-canary-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let fortran_vb = |name: &str, weight: &Tensor| {
            let path = dir.join(format!("{name}.pth"));
            write_f32_state_dict(&path, "weight", weight, true);
            VarBuilder::from_pth(path, DType::F32, &cpu).unwrap()
        };
        let mlx_vb = |weight: &Tensor| {
            let weights = std::collections::HashMap::from([("weight".to_string(), weight.clone())]);
            VarBuilder::from_tensors(weights, DType::F32, &cpu)
        };

        // Candle's own `.pth` reader returns a Fortran tensor as a strided view.
        let raw = fortran_vb("raw", &oik).get((6, 8, 3), "weight").unwrap();
        assert!(!raw.is_contiguous());

        let mut failures = Vec::new();
        let conv1d_layouts = [
            ("oki", mlx_vb(&oki)),
            ("fortran-oik", fortran_vb("oik", &oik)),
        ];
        for (layout, vb) in conv1d_layouts {
            for (loader, conv) in [
                (
                    "load_conv1d",
                    load_conv1d(8, 6, 3, cfg, vb.clone()).unwrap(),
                ),
                (
                    "load_conv1d_no_bias",
                    load_conv1d_no_bias(8, 6, 3, cfg, vb).unwrap(),
                ),
            ] {
                let err = max_abs_diff(&host_f32(&conv.forward(&input).unwrap()).unwrap(), &want);
                if !conv.weight().is_contiguous() || err > 1e-6 {
                    failures.push(format!("{loader} {layout}: max|d|={err:e}"));
                }
            }
        }

        let ohwi = xorshift_tensor(&[6, 3, 3, 8], 13, 0.05, &cpu).unwrap();
        let oihw = ohwi.permute((0, 3, 1, 2)).unwrap().contiguous().unwrap();
        let image = xorshift_tensor(&[1, 8, 9, 7], 17, 1.0, &cpu).unwrap();
        let cfg2d = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let want2d = host_f32(
            &Conv2d::new(oihw.clone(), None, cfg2d)
                .forward(&image)
                .unwrap(),
        )
        .unwrap();
        let conv2d_layouts = [
            ("ohwi", mlx_vb(&ohwi)),
            ("fortran-oihw", fortran_vb("oihw", &oihw)),
        ];
        for (layout, vb) in conv2d_layouts {
            let conv = load_conv2d(8, 6, 3, cfg2d, vb).unwrap();
            let err = max_abs_diff(&host_f32(&conv.forward(&image).unwrap()).unwrap(), &want2d);
            if !conv.weight().is_contiguous() || err > 1e-6 {
                failures.push(format!("load_conv2d {layout}: max|d|={err:e}"));
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
        assert!(failures.is_empty(), "strided conv kernels: {failures:?}");
    }
}

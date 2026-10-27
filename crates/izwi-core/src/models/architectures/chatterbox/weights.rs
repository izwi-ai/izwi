//! Tensor loading helpers shared by Chatterbox's modules.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

use crate::error::{Error, Result};
use crate::models::shared::weights::staged::metal_staged_safetensors;

/// A memory-mapped safetensors file as a `VarBuilder` producing `dtype` on
/// `device`, staged on Metal: a tensor stored in another dtype is converted
/// on the host and uploaded at its exact size, so a lowered T3 never holds
/// its F32 upload beside the cast.
pub(crate) fn mmap_safetensors<'a>(
    path: &Path,
    dtype: DType,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    if !path.is_file() {
        return Err(Error::ModelLoadError(format!(
            "Chatterbox weights {} are missing",
            path.display()
        )));
    }
    // SAFETY: the file is a read-only model artifact izwi verified on download
    // and does not modify while loaded.
    unsafe { metal_staged_safetensors(&[path], dtype, device) }.map_err(|error| {
        Error::ModelLoadError(format!(
            "Failed to map Chatterbox weights {}: {error}",
            path.display()
        ))
    })
}

/// `g * v / ||v||` over every dimension but the first, from PyTorch's
/// `parametrizations.weight.original0` (g) and `original1` (v). Fused on the
/// host in F32 and uploaded once at `dtype`.
pub(crate) fn weight_norm_kernel(vb: &VarBuilder, dtype: DType) -> Result<Tensor> {
    let host = vb.clone().set_device(Device::Cpu).set_dtype(DType::F32);
    let g = host.get_unchecked_dtype("parametrizations.weight.original0", DType::F32)?;
    let v = host.get_unchecked_dtype("parametrizations.weight.original1", DType::F32)?;
    let rank = v.rank();
    if rank < 2 || g.dims()[0] != v.dims()[0] {
        return Err(Error::ModelLoadError(format!(
            "Chatterbox weight-norm pair has shapes {:?} and {:?}",
            g.dims(),
            v.dims()
        )));
    }
    let reduce: Vec<usize> = (1..rank).collect();
    let norm = v.sqr()?.sum_keepdim(reduce)?.sqrt()?;
    let kernel = v.broadcast_mul(&g.broadcast_div(&norm)?)?;
    Ok(kernel
        .to_dtype(dtype)?
        .to_device(vb.device())?
        .contiguous()?)
}

fn optional_bias(vb: &VarBuilder, dtype: DType) -> Result<Option<Tensor>> {
    if vb.contains_tensor("bias") {
        Ok(Some(vb.get_unchecked_dtype("bias", dtype)?))
    } else {
        Ok(None)
    }
}

pub(crate) fn conv1d(vb: &VarBuilder, cfg: Conv1dConfig) -> Result<Conv1d> {
    let dtype = vb.dtype();
    let weight = vb.get_unchecked_dtype("weight", dtype)?.contiguous()?;
    Ok(Conv1d::new(weight, optional_bias(vb, dtype)?, cfg))
}

pub(crate) fn weight_norm_conv1d(vb: &VarBuilder, cfg: Conv1dConfig) -> Result<Conv1d> {
    let dtype = vb.dtype();
    Ok(Conv1d::new(
        weight_norm_kernel(vb, dtype)?,
        optional_bias(vb, dtype)?,
        cfg,
    ))
}

pub(crate) fn weight_norm_conv_transpose1d(
    vb: &VarBuilder,
    cfg: ConvTranspose1dConfig,
) -> Result<ConvTranspose1d> {
    let dtype = vb.dtype();
    Ok(ConvTranspose1d::new(
        weight_norm_kernel(vb, dtype)?,
        optional_bias(vb, dtype)?,
        cfg,
    ))
}

pub(crate) fn conv_cfg(padding: usize, stride: usize, dilation: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding,
        stride,
        dilation,
        groups: 1,
        cudnn_fwd_algo: None,
    }
}

/// `[rows, cols]` row-major F32 values of a host tensor.
pub(crate) fn host_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?)
}

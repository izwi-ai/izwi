//! Minimal PyTorch zip/pickle tensor loading helpers.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::{DType, Device, Tensor};
use zip::ZipArchive;

use crate::error::{Error, Result};

pub fn read_single_tensor_pth(
    path: &Path,
    device: &Device,
    dtype: DType,
    logical_name: &str,
) -> Result<Tensor> {
    let (tensor_info, archive_member_path) = read_single_tensor_info(path, logical_name)?;
    let tensor = read_tensor_from_zip(path, &tensor_info, &archive_member_path)?;
    let tensor = if tensor.dtype() != dtype {
        tensor.to_dtype(dtype)?
    } else {
        tensor
    };
    if tensor.device().same_device(device) {
        Ok(tensor)
    } else {
        tensor.to_device(device).map_err(Error::from)
    }
}

pub fn read_single_tensor_info(path: &Path, logical_name: &str) -> Result<(TensorInfo, String)> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to open PyTorch tensor archive {}: {}",
            path.display(),
            err
        ))
    })?;

    let file_names = zip.file_names().map(str::to_string).collect::<Vec<_>>();
    for file_name in &file_names {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = PathBuf::from(
            file_name
                .strip_suffix(".pkl")
                .ok_or_else(|| Error::ModelLoadError("Invalid PyTorch pickle path".to_string()))?,
        );
        let reader = zip.by_name(file_name).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to read pickle payload {} from {}: {}",
                file_name,
                path.display(),
                err
            ))
        })?;
        let mut reader = BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader).map_err(Error::from)?;
        let obj = stack.finalize().map_err(Error::from)?;
        let info = obj
            .into_tensor_info(Object::Unicode(logical_name.to_string()), &dir_name)
            .map_err(Error::from)?
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "PyTorch archive {} did not contain a top-level tensor",
                    path.display()
                ))
            })?;
        return Ok((info.clone(), info.path.clone()));
    }

    Err(Error::ModelLoadError(format!(
        "Could not find data.pkl inside PyTorch tensor archive {}",
        path.display()
    )))
}

fn read_tensor_from_zip(path: &Path, info: &TensorInfo, member_path: &str) -> Result<Tensor> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to open PyTorch tensor archive {}: {}",
            path.display(),
            err
        ))
    })?;
    let mut reader = zip.by_name(member_path).map_err(|err| {
        Error::ModelLoadError(format!(
            "Missing tensor storage member {} in {}: {}",
            member_path,
            path.display(),
            err
        ))
    })?;

    let is_fortran = info.layout.is_fortran_contiguous();
    let rank = info.layout.shape().rank();
    if !info.layout.is_contiguous() && !is_fortran {
        return Err(Error::ModelLoadError(format!(
            "Unsupported non-contiguous PyTorch tensor layout {:?} in {}",
            info.layout,
            path.display()
        )));
    }

    let start_offset = info.layout.start_offset();
    if start_offset > 0 {
        std::io::copy(
            &mut reader.by_ref().take(start_offset as u64),
            &mut std::io::sink(),
        )?;
    }

    let elem_count = info.layout.shape().elem_count();
    let byte_len = elem_count
        .checked_mul(info.dtype.size_in_bytes())
        .ok_or_else(|| {
            Error::ModelLoadError("PyTorch tensor byte length overflowed".to_string())
        })?;
    let mut raw = vec![0u8; byte_len];
    reader.read_exact(&mut raw)?;

    let tensor = if rank > 1 && is_fortran {
        let reversed_shape = info.layout.dims().iter().rev().copied().collect::<Vec<_>>();
        let tensor = Tensor::from_raw_buffer(&raw, info.dtype, &reversed_shape, &Device::Cpu)?;
        let perm = (0..rank).rev().collect::<Vec<_>>();
        tensor.permute(perm)?
    } else {
        Tensor::from_raw_buffer(&raw, info.dtype, info.layout.dims(), &Device::Cpu)?
    };

    Ok(tensor)
}

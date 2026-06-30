//! Minimal PyTorch zip/pickle tensor loading helpers.

use std::collections::BTreeMap;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::{DType, Device, Tensor};
use zip::ZipArchive;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PthTensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub archive_member_path: String,
}

#[derive(Debug, Clone)]
pub struct PthTensorMap {
    path: PathBuf,
    tensor_infos: BTreeMap<String, TensorInfo>,
    selected_key: Option<String>,
}

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

impl PthTensorMap {
    pub fn open(path: &Path, key: Option<&str>) -> Result<Self> {
        let tensor_infos = candle_core::pickle::read_pth_tensor_info(path, false, key)
            .map_err(Error::from)?
            .into_iter()
            .map(|info| (info.name.clone(), info))
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            path: path.to_path_buf(),
            tensor_infos,
            selected_key: key.map(str::to_string),
        })
    }

    pub fn open_first_non_empty(path: &Path, keys: &[Option<&str>]) -> Result<Self> {
        let mut errors = Vec::new();
        for key in keys {
            match Self::open(path, *key) {
                Ok(map) if !map.is_empty() => return Ok(map),
                Ok(_) => {}
                Err(err) => errors.push(format!("key {key:?}: {err}")),
            }
        }
        let detail = if errors.is_empty() {
            "all candidate keys were empty".to_string()
        } else {
            errors.join("; ")
        };
        Err(Error::ModelLoadError(format!(
            "PyTorch archive {} did not contain named tensors ({detail})",
            path.display()
        )))
    }

    pub fn selected_key(&self) -> Option<&str> {
        self.selected_key.as_deref()
    }

    pub fn len(&self) -> usize {
        self.tensor_infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensor_infos.is_empty()
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.tensor_infos.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensor_infos.keys().map(String::as_str)
    }

    pub fn specs(&self) -> Vec<PthTensorSpec> {
        self.tensor_infos
            .values()
            .map(|info| PthTensorSpec {
                name: info.name.clone(),
                dtype: info.dtype,
                shape: info.layout.dims().to_vec(),
                archive_member_path: info.path.clone(),
            })
            .collect()
    }

    pub fn get(&self, name: &str, device: &Device, dtype: Option<DType>) -> Result<Option<Tensor>> {
        let Some(info) = self.tensor_infos.get(name) else {
            return Ok(None);
        };
        let mut tensor = read_tensor_from_zip(&self.path, info, &info.path)?;
        if let Some(dtype) = dtype {
            if tensor.dtype() != dtype {
                tensor = tensor.to_dtype(dtype)?;
            }
        }
        if !tensor.device().same_device(device) {
            tensor = tensor.to_device(device)?;
        }
        Ok(Some(tensor))
    }

    pub fn read_all(
        &self,
        device: &Device,
        dtype: Option<DType>,
    ) -> Result<BTreeMap<String, Tensor>> {
        let mut tensors = BTreeMap::new();
        for name in self.tensor_infos.keys() {
            let tensor = self.get(name, device, dtype)?.ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "PyTorch archive tensor disappeared while reading: {name}"
                ))
            })?;
            tensors.insert(name.clone(), tensor);
        }
        Ok(tensors)
    }
}

//! Minimal PyTorch zip/pickle tensor loading helpers.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::{DType, Device, Tensor};
use zip::read::{ArchiveOffset, Config};
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
    let mut zip = open_archive(path)?;
    let (tensor_info, archive_member_path) = read_tensor_info(&mut zip, path, logical_name)?;
    let tensor = read_tensor_from_zip(&mut zip, path, &tensor_info, &archive_member_path)?;
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
    read_tensor_info(&mut open_archive(path)?, path, logical_name)
}

type PthArchive = ZipArchive<BufReader<File>>;

/// `torch.save` writes the archive from byte 0, so the offset is given rather
/// than detected: zip 9's detection scans the whole file backwards, about 2 s
/// per open on a 1.9 GB checkpoint. An archive with bytes in front of it fails
/// the fixed offset and is reopened with detection.
fn open_archive(path: &Path) -> Result<PthArchive> {
    let at_start = Config {
        archive_offset: ArchiveOffset::Known(0),
    };
    if let Ok(zip) = ZipArchive::with_config(at_start, BufReader::new(File::open(path)?)) {
        return Ok(zip);
    }
    ZipArchive::new(BufReader::new(File::open(path)?)).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to open PyTorch tensor archive {}: {}",
            path.display(),
            err
        ))
    })
}

fn read_tensor_info(
    zip: &mut PthArchive,
    path: &Path,
    logical_name: &str,
) -> Result<(TensorInfo, String)> {
    let file_names = zip
        .file_names()
        .map(|name| name.map(|name| name.into_owned()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to list PyTorch tensor archive {} entries: {}",
                path.display(),
                err
            ))
        })?;
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

fn read_tensor_from_zip(
    zip: &mut PthArchive,
    path: &Path,
    info: &TensorInfo,
    member_path: &str,
) -> Result<Tensor> {
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
        // A permuted view would reach conv kernels strided; Candle 0.11's
        // im2col convs multiply a strided kernel as if it were row-major.
        tensor.permute(perm)?.contiguous()?
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
        self.read_from(&mut open_archive(&self.path)?, info, device, dtype)
            .map(Some)
    }

    fn read_from(
        &self,
        zip: &mut PthArchive,
        info: &TensorInfo,
        device: &Device,
        dtype: Option<DType>,
    ) -> Result<Tensor> {
        let mut tensor = read_tensor_from_zip(zip, &self.path, info, &info.path)?;
        if let Some(dtype) = dtype {
            if tensor.dtype() != dtype {
                tensor = tensor.to_dtype(dtype)?;
            }
        }
        if !tensor.device().same_device(device) {
            tensor = tensor.to_device(device)?;
        }
        Ok(tensor)
    }

    pub fn read_all(
        &self,
        device: &Device,
        dtype: Option<DType>,
    ) -> Result<BTreeMap<String, Tensor>> {
        self.read_where(device, dtype, |_| true)
    }

    /// [`Self::read_all`] for the tensors whose name `keep` accepts; the
    /// others are never read.
    pub fn read_where(
        &self,
        device: &Device,
        dtype: Option<DType>,
        keep: impl Fn(&str) -> bool,
    ) -> Result<BTreeMap<String, Tensor>> {
        let mut zip = open_archive(&self.path)?;
        let mut tensors = BTreeMap::new();
        for (name, info) in self.tensor_infos.iter().filter(|(name, _)| keep(name)) {
            let tensor = self.read_from(&mut zip, info, device, dtype)?;
            tensors.insert(name.clone(), tensor);
        }
        Ok(tensors)
    }
}

/// Torch-free `.pth` fixtures for loader tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::io::Write;
    use std::path::Path;

    use candle_core::Tensor;

    fn unicode(pickle: &mut Vec<u8>, value: &str) {
        pickle.push(b'X');
        pickle.extend_from_slice(&(value.len() as u32).to_le_bytes());
        pickle.extend_from_slice(value.as_bytes());
    }

    fn global(pickle: &mut Vec<u8>, module: &str, name: &str) {
        pickle.push(b'c');
        pickle.extend_from_slice(format!("{module}\n{name}\n").as_bytes());
    }

    fn int_tuple(pickle: &mut Vec<u8>, values: &[usize]) {
        pickle.push(b'(');
        for &value in values {
            pickle.push(b'J');
            pickle.extend_from_slice(&(value as i32).to_le_bytes());
        }
        pickle.push(b't');
    }

    /// Write `tensor` (F32) as the one-entry state dict `{name: tensor}`, in
    /// the zip/pickle layout `torch.save` produces. With `fortran`, the
    /// storage is column-major, as for a tensor saved from a transposed view.
    pub(crate) fn write_f32_state_dict(path: &Path, name: &str, tensor: &Tensor, fortran: bool) {
        let dims = tensor.dims().to_vec();
        let rank = dims.len();
        let (storage, strides) = if fortran {
            let reversed = (0..rank).rev().collect::<Vec<_>>();
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1;
            for &dim in &dims {
                strides.push(stride);
                stride *= dim;
            }
            (
                tensor.permute(reversed).unwrap().contiguous().unwrap(),
                strides,
            )
        } else {
            (tensor.contiguous().unwrap(), tensor.stride().to_vec())
        };
        let values = storage.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut pickle = vec![0x80, 0x02, b'}'];
        unicode(&mut pickle, name);
        global(&mut pickle, "torch._utils", "_rebuild_tensor_v2");
        pickle.push(b'(');
        pickle.push(b'(');
        unicode(&mut pickle, "storage");
        global(&mut pickle, "torch", "FloatStorage");
        unicode(&mut pickle, "0");
        unicode(&mut pickle, "cpu");
        pickle.push(b'J');
        pickle.extend_from_slice(&(values.len() as i32).to_le_bytes());
        pickle.extend_from_slice(b"tQ");
        pickle.extend_from_slice(&[b'K', 0]);
        int_tuple(&mut pickle, &dims);
        int_tuple(&mut pickle, &strides);
        pickle.push(0x89);
        global(&mut pickle, "collections", "OrderedDict");
        pickle.extend_from_slice(b")RtRs.");

        let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("archive/data.pkl", options).unwrap();
        zip.write_all(&pickle).unwrap();
        zip.start_file("archive/data/0", options).unwrap();
        for value in values {
            zip.write_all(&value.to_le_bytes()).unwrap();
        }
        zip.finish().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fortran_ordered_pth_tensor_loads_contiguous() {
        let dir = std::env::temp_dir().join(format!("izwi-pth-fortran-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let reference = Tensor::arange(0f32, 24., &Device::Cpu)
            .unwrap()
            .reshape((2, 3, 4))
            .unwrap();
        for fortran in [false, true] {
            let path = dir.join(format!("w-{fortran}.pth"));
            test_support::write_f32_state_dict(&path, "w", &reference, fortran);
            let map = PthTensorMap::open(&path, None).unwrap();
            let info = &map.tensor_infos["w"];
            assert_eq!(info.layout.is_fortran_contiguous(), fortran);

            let loaded = map.get("w", &Device::Cpu, None).unwrap().unwrap();
            assert!(loaded.is_contiguous(), "fortran={fortran}");
            assert_eq!(
                loaded.to_vec3::<f32>().unwrap(),
                reference.to_vec3::<f32>().unwrap(),
                "fortran={fortran}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn archive_with_leading_bytes_still_loads() {
        let dir = std::env::temp_dir().join(format!("izwi-pth-offset-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let reference = Tensor::arange(0f32, 12., &Device::Cpu)
            .unwrap()
            .reshape((3, 4))
            .unwrap();
        let plain = dir.join("plain.pth");
        test_support::write_f32_state_dict(&plain, "w", &reference, false);
        let shifted = dir.join("shifted.pth");
        let mut bytes = vec![0x5a; 4099];
        bytes.extend(std::fs::read(&plain).unwrap());
        std::fs::write(&shifted, bytes).unwrap();
        assert!(ZipArchive::with_config(
            Config {
                archive_offset: ArchiveOffset::Known(0)
            },
            BufReader::new(File::open(&shifted).unwrap())
        )
        .is_err());

        for path in [&plain, &shifted] {
            let map = PthTensorMap::open(path, None).unwrap();
            let one = map.get("w", &Device::Cpu, None).unwrap().unwrap();
            let all = map.read_all(&Device::Cpu, None).unwrap();
            assert_eq!(all.len(), 1);
            for tensor in [&one, &all["w"]] {
                assert_eq!(
                    tensor.to_vec2::<f32>().unwrap(),
                    reference.to_vec2::<f32>().unwrap(),
                    "{}",
                    path.display()
                );
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}

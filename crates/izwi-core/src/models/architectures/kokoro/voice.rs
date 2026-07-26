use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::{DType, IndexOp, Tensor};
use zip::ZipArchive;

use crate::error::{Error, Result};

pub(crate) const KOKORO_MODEL_MEMO_MAX_BYTES: u64 = (510 * 256 * std::mem::size_of::<f32>()) as u64;

#[derive(Debug)]
pub struct VoiceLibrary {
    voices_dir: PathBuf,
    device: candle_core::Device,
    dtype: DType,
    cache: RwLock<Option<(String, Tensor)>>,
}

impl VoiceLibrary {
    pub fn new(voices_dir: PathBuf, device: candle_core::Device, dtype: DType) -> Result<Self> {
        if !voices_dir.exists() {
            return Err(Error::ModelNotFound(format!(
                "Kokoro voices directory not found at {}",
                voices_dir.display()
            )));
        }
        Ok(Self {
            voices_dir,
            device,
            dtype,
            cache: RwLock::new(None),
        })
    }

    pub fn list_speakers(&self) -> Result<Vec<String>> {
        let mut speakers = Vec::new();
        for entry in std::fs::read_dir(&self.voices_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("pt") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                speakers.push(stem.to_string());
            }
        }
        speakers.sort();
        Ok(speakers)
    }

    pub fn load_pack(&self, speaker: &str) -> Result<Tensor> {
        if let Some(cached) = self
            .cache
            .read()
            .map_err(|_| Error::ModelLoadError("Voice cache lock poisoned".to_string()))?
            .as_ref()
            .filter(|(cached_speaker, _)| cached_speaker == speaker)
            .map(|(_, tensor)| tensor.clone())
        {
            return Ok(cached);
        }

        let path = self.voices_dir.join(format!("{speaker}.pt"));
        if !path.exists() {
            return Err(Error::InvalidInput(format!(
                "Unknown Kokoro speaker '{speaker}' (missing voice pack {})",
                path.display()
            )));
        }
        let tensor = read_single_tensor_pth(&path, &self.device, self.dtype)?;
        let dims = tensor.shape().dims().to_vec();
        if dims.as_slice() != [510, 1, 256] {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Kokoro voice pack shape for {}: {:?} (expected [510,1,256])",
                path.display(),
                dims
            )));
        }

        self.cache
            .write()
            .map_err(|_| Error::ModelLoadError("Voice cache lock poisoned".to_string()))?
            .replace((speaker.to_string(), tensor.clone()));
        Ok(tensor)
    }

    pub fn style_for_phoneme_len(&self, speaker: &str, phoneme_len: usize) -> Result<Tensor> {
        let pack = self.load_pack(speaker)?;
        let clamped_len = phoneme_len.clamp(1, 510);
        let idx = clamped_len - 1;
        pack.i((idx, .., ..)).map_err(Error::from)
    }
}

fn read_single_tensor_pth(
    path: &Path,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let (tensor_info, archive_member_path) = read_single_tensor_info(path)?;
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

fn read_single_tensor_info(path: &Path) -> Result<(TensorInfo, String)> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to open Kokoro voice pack {} as zip archive: {}",
            path.display(),
            e
        ))
    })?;

    let file_names = zip.file_names().map(|s| s.to_string()).collect::<Vec<_>>();
    for file_name in &file_names {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = std::path::PathBuf::from(
            file_name
                .strip_suffix(".pkl")
                .ok_or_else(|| Error::ModelLoadError("Invalid voice pickle path".to_string()))?,
        );
        let reader = zip.by_name(file_name).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to read voice pickle payload {} from {}: {}",
                file_name,
                path.display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader).map_err(Error::from)?;
        let obj = stack.finalize().map_err(Error::from)?;
        let info = obj
            .into_tensor_info(Object::Unicode("__voice_pack__".to_string()), &dir_name)
            .map_err(Error::from)?
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Kokoro voice pack {} did not contain a top-level tensor",
                    path.display()
                ))
            })?;
        return Ok((info.clone(), info.path.clone()));
    }

    Err(Error::ModelLoadError(format!(
        "Could not find data.pkl inside Kokoro voice pack {}",
        path.display()
    )))
}

fn read_tensor_from_zip(path: &Path, info: &TensorInfo, member_path: &str) -> Result<Tensor> {
    use std::io::Read;

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to open Kokoro voice pack {}: {}",
            path.display(),
            e
        ))
    })?;
    let mut reader = zip.by_name(member_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Missing tensor storage member {} in {}: {}",
            member_path,
            path.display(),
            e
        ))
    })?;

    let is_fortran = info.layout.is_fortran_contiguous();
    let rank = info.layout.shape().rank();
    if !info.layout.is_contiguous() && !is_fortran {
        return Err(Error::ModelLoadError(format!(
            "Unsupported non-contiguous Kokoro voice tensor layout {:?}",
            info.layout
        )));
    }

    let start_offset = info.layout.start_offset();
    if start_offset > 0 {
        std::io::copy(
            &mut reader.by_ref().take(start_offset as u64),
            &mut std::io::sink(),
        )?;
    }

    if info.dtype != DType::F32 {
        return Err(Error::ModelLoadError(format!(
            "Unsupported Kokoro voice tensor dtype {:?}; expected F32",
            info.dtype
        )));
    }

    let elem_count = info.layout.shape().elem_count();
    let byte_len = elem_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            Error::ModelLoadError("Kokoro voice tensor byte length overflow".to_string())
        })?;
    let mut raw = vec![0u8; byte_len];
    reader.read_exact(&mut raw)?;
    let mut data = Vec::with_capacity(elem_count);
    for chunk in raw.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let tensor = Tensor::from_vec(data, info.layout.shape().clone(), &candle_core::Device::Cpu)?;
    if rank > 1 && is_fortran {
        let reversed_shape: Vec<_> = info.layout.dims().iter().rev().copied().collect();
        let tensor = tensor.reshape(reversed_shape)?;
        let perm: Vec<_> = (0..rank).rev().collect();
        tensor.permute(perm).map_err(Error::from)
    } else {
        Ok(tensor)
    }
}

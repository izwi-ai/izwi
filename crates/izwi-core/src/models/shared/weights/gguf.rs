//! GGUF model loader for quantized model support.
//!
//! This module provides functionality to load models in the GGUF (GPT-Generated Unified Format)
//! format, which is commonly used for quantized LLMs. GGUF supports various quantization
//! schemes including Q4_K_M, Q5_K_M, Q8_0, and others.
//!
//! The loader integrates with Candle's existing VarBuilder pattern, allowing models
//! to be loaded from either safetensors or GGUF formats with minimal code changes.

use candle_core::quantized::gguf_file::{Content as GgufContent, Value as GgufValue};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::backends::{backend_kind_for_device, open_gguf_reader, BackendKind};
use crate::error::{Error, Result};

/// Information about a GGUF model file.
#[derive(Debug, Clone)]
pub struct GgufModelInfo {
    /// Path to the GGUF file
    pub path: PathBuf,
    /// Model architecture (if specified in metadata)
    pub architecture: Option<String>,
    /// Model size parameters
    pub parameter_count: Option<u64>,
    /// Quantization type (e.g., "Q4_K_M", "Q5_K_M", "Q8_0")
    pub quantization: Option<String>,
    /// General metadata
    pub metadata: HashMap<String, String>,
}

/// GGUF model loader that provides a unified interface for loading quantized models.
pub struct GgufLoader {
    path: PathBuf,
    content: GgufContent,
    tensor_count: usize,
    metadata: HashMap<String, GgufValue>,
}

impl GgufLoader {
    /// Create a new GGUF loader from a file path.
    pub fn from_path(path: &Path) -> Result<Self> {
        Self::from_path_with_backend(path, BackendKind::Cpu)
    }

    pub fn from_path_with_backend(path: &Path, backend: BackendKind) -> Result<Self> {
        let mut reader = open_gguf_reader(path, backend)
            .map_err(|e| Error::ModelLoadError(format!("Failed to open GGUF file: {}", e)))?;
        let content = GgufContent::read(&mut reader)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse GGUF file: {}", e)))?;

        let tensor_count = content.tensor_infos.len();
        let metadata: HashMap<String, GgufValue> = content.metadata.clone().into_iter().collect();

        tracing::info!(
            "Loaded GGUF file with {} tensors and {} metadata entries",
            tensor_count,
            metadata.len()
        );

        Ok(Self {
            path: path.to_path_buf(),
            content,
            tensor_count,
            metadata,
        })
    }

    /// Get model information from the GGUF metadata.
    pub fn get_model_info(&self) -> GgufModelInfo {
        let architecture = self
            .get_metadata_string("general.architecture")
            .or_else(|| self.get_metadata_string("general.name"));

        let parameter_count = self.get_metadata_u64("general.parameter_count");

        let quantization = self
            .get_metadata_string("general.quantization_type")
            .or_else(|| self.detect_quantization_from_tensors());

        let mut metadata = HashMap::new();
        for (key, value) in &self.metadata {
            if let Ok(s) = format_gguf_value(value) {
                metadata.insert(key.clone(), s);
            }
        }

        GgufModelInfo {
            path: self.path.clone(),
            architecture,
            parameter_count,
            quantization,
            metadata,
        }
    }

    /// Check if a tensor exists in the GGUF file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    /// Get the logical shape of a tensor without loading its data.
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.content
            .tensor_infos
            .get(name)
            .map(|info| info.shape.dims().to_vec())
    }

    /// Load a quantized tensor from the GGUF file.
    ///
    /// Delegates to Candle's `Content::tensor`, which fully supports
    /// all GGUF dtypes (F32/F16 and quantized families like Q4_K/Q5_K).
    pub fn load_qtensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        let mut reader = open_gguf_reader(&self.path, backend_kind_for_device(device))
            .map_err(|e| Error::ModelLoadError(format!("Failed to reopen GGUF file: {}", e)))?;

        self.content.tensor(&mut reader, name, device).map_err(|e| {
            Error::ModelLoadError(format!("Failed to load tensor '{}' from GGUF: {}", name, e))
        })
    }

    /// Load a tensor, automatically dequantizing if necessary.
    pub fn load_tensor(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let qtensor = self.load_qtensor(name, device)?;

        // Dequantize to the requested dtype
        let tensor = qtensor.dequantize(device).map_err(|e| {
            Error::ModelLoadError(format!("Failed to dequantize tensor '{}': {}", name, e))
        })?;

        // Cast to requested dtype if different
        if tensor.dtype() != dtype {
            tensor.to_dtype(dtype).map_err(|e| {
                Error::ModelLoadError(format!("Failed to cast tensor '{}': {}", name, e))
            })
        } else {
            Ok(tensor)
        }
    }

    /// Get all tensor names in the GGUF file.
    pub fn tensor_names(&self) -> Vec<String> {
        self.content.tensor_infos.keys().cloned().collect()
    }

    /// Number of tensors present in the GGUF checkpoint.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Get a raw metadata value.
    pub fn metadata_value(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a string metadata value.
    pub fn get_metadata_string(&self, key: &str) -> Option<String> {
        self.metadata.get(key).and_then(|v| {
            if let GgufValue::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    /// Get a u64 metadata value.
    pub fn get_metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| match v {
            GgufValue::U64(n) => Some(*n),
            GgufValue::I64(n) => Some(*n as u64),
            GgufValue::U32(n) => Some(*n as u64),
            _ => None,
        })
    }

    /// Detect quantization type by examining tensor dtypes.
    fn detect_quantization_from_tensors(&self) -> Option<String> {
        let mut dtypes = std::collections::HashSet::new();
        for info in self.content.tensor_infos.values() {
            dtypes.insert(format!("{:?}", info.ggml_dtype));
        }

        // Return the most common quantized dtype
        let quant_types: Vec<_> = dtypes
            .iter()
            .filter(|d| !d.contains("F32") && !d.contains("F16"))
            .cloned()
            .collect();

        if quant_types.len() == 1 {
            Some(quant_types[0].clone())
        } else if !quant_types.is_empty() {
            Some(format!("mixed:{}", quant_types.join(",")))
        } else {
            None
        }
    }
}

/// Load a model from either GGUF or safetensors format.
///
/// Automatically detects the file format based on magic bytes and loads
/// the model weights accordingly. Supports:
/// - GGUF format (quantized models)
/// - Safetensors format (standard HuggingFace format)
///
/// # Arguments
/// * `path` - Path to the model file (or directory for safetensors)
/// * `dtype` - Target data type for the tensors
/// * `device` - Device to load the tensors on
///
/// # Returns
/// A VarBuilder containing all loaded tensors
pub fn load_model_weights(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    if is_gguf_file(path) {
        tracing::info!("Detected GGUF format, loading from {:?}", path);
        var_builder_from_gguf(path, dtype, device)
    } else {
        tracing::info!("Detected safetensors format, loading from {:?}", path);
        var_builder_from_safetensors(path, dtype, device)
    }
}

/// Create a VarBuilder from safetensors file(s).
///
/// Supports both single-file and sharded safetensors models.
fn var_builder_from_safetensors(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // Check if this is a directory or single file
    if path.is_dir() {
        // Look for model.safetensors or sharded files
        let model_path = path.join("model.safetensors");
        let index_path = path.join("model.safetensors.index.json");

        if model_path.exists() {
            // Single file
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path.clone()], dtype, device).map_err(
                    |e| Error::ModelLoadError(format!("Failed to load safetensors: {}", e)),
                )?
            };
            Ok(vb)
        } else if index_path.exists() {
            // Sharded model - read index and load all shards
            let index_data = std::fs::read_to_string(&index_path)
                .map_err(|e| Error::ModelLoadError(format!("Failed to read index: {}", e)))?;
            let index: serde_json::Value = serde_json::from_str(&index_data)
                .map_err(|e| Error::ModelLoadError(format!("Failed to parse index: {}", e)))?;

            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| Error::ModelLoadError("Invalid index format".to_string()))?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> =
                shard_files.iter().map(|f| path.join(f)).collect();

            tracing::info!("Loading sharded model with {} shards", shard_paths.len());

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, device)
                    .map_err(|e| Error::ModelLoadError(format!("Failed to load shards: {}", e)))?
            };
            Ok(vb)
        } else {
            Err(Error::ModelNotFound(format!(
                "No model files found in {:?}",
                path
            )))
        }
    } else {
        // Single file path
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.to_path_buf()], dtype, device)
                .map_err(|e| Error::ModelLoadError(format!("Failed to load safetensors: {}", e)))?
        };
        Ok(vb)
    }
}

/// Create a VarBuilder from a GGUF file.
///
/// This allows loading models from GGUF format using the same interface
/// as safetensors files.
pub fn var_builder_from_gguf(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let loader = GgufLoader::from_path_with_backend(path, backend_kind_for_device(device))?;

    // Load all tensors into a HashMap
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    for name in loader.tensor_names() {
        match loader.load_tensor(&name, dtype, device) {
            Ok(tensor) => {
                tensors.insert(name, tensor);
            }
            Err(e) => {
                tracing::warn!("Failed to load tensor '{}': {}", name, e);
            }
        }
    }

    tracing::info!("Loaded {} tensors from GGUF file", tensors.len());

    // Create VarBuilder from the tensors
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    Ok(vb)
}

/// Check if a file is in GGUF format by examining the magic bytes.
pub fn is_gguf_file(path: &Path) -> bool {
    if let Ok(mut file) = std::fs::File::open(path) {
        use std::io::Read;
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() {
            // GGUF magic: 'GGUF' in little-endian
            return magic == [0x47, 0x47, 0x55, 0x46]; // "GGUF"
        }
    }
    false
}

/// Helper function to format GGUF metadata values for display.
fn format_gguf_value(value: &GgufValue) -> Result<String> {
    match value {
        GgufValue::String(s) => Ok(s.clone()),
        GgufValue::U64(n) => Ok(n.to_string()),
        GgufValue::I64(n) => Ok(n.to_string()),
        GgufValue::F64(n) => Ok(n.to_string()),
        GgufValue::Bool(b) => Ok(b.to_string()),
        GgufValue::Array(arr) => {
            let len = arr.len();
            Ok(format!("[{} elements]", len))
        }
        GgufValue::U32(n) => Ok(n.to_string()),
        GgufValue::I32(n) => Ok(n.to_string()),
        GgufValue::F32(n) => Ok(n.to_string()),
        GgufValue::U8(n) => Ok(n.to_string()),
        GgufValue::I8(n) => Ok(n.to_string()),
        GgufValue::U16(n) => Ok(n.to_string()),
        GgufValue::I16(n) => Ok(n.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_detection() {
        // Create a temporary file with GGUF magic
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test.gguf");

        std::fs::write(&test_file, b"GGUF\x00\x00\x00\x00").unwrap();
        assert!(is_gguf_file(&test_file));

        // Clean up
        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_non_gguf_file() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test.txt");

        std::fs::write(&test_file, b"This is not a GGUF file").unwrap();
        assert!(!is_gguf_file(&test_file));

        // Clean up
        let _ = std::fs::remove_file(&test_file);
    }
}

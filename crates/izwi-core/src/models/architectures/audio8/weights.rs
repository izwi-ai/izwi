//! Audio8 safetensor loading: one `model.safetensors`, flat tensor names.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::backends::{BackendKind, DeviceKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::models::architectures::audio8::artifacts::AUDIO8_WEIGHTS_FILE;
use crate::models::architectures::audio8::contracts::remap_audio8_key;
use crate::models::shared::config::checkpoint_dtype_from_config_json;
use crate::models::shared::weights::pytorch::PthTensorMap;

use super::codec::{audio8_codec_memory, Audio8CodecMemory};
use super::config::Audio8Config;
use super::dac::Audio8DacConfig;
use super::fast::Audio8FastConfig;
use super::slow::Audio8SlowConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio8TensorSpec {
    pub source_name: String,
    pub remapped_name: String,
    pub shard_file: String,
    pub shape: Vec<usize>,
    pub source_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Audio8WeightIndex {
    model_dir: PathBuf,
    shard_files: Vec<String>,
    tensors: BTreeMap<String, Audio8TensorSpec>,
    source_to_remapped: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct Audio8Weights {
    dtype: DType,
    device: Device,
    index: Audio8WeightIndex,
    vb: VarBuilder<'static>,
}

impl Audio8WeightIndex {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let shard_file = AUDIO8_WEIGHTS_FILE.to_string();
        let shard_path = model_dir.join(&shard_file);
        // Inspect the validated safetensors header without copying the
        // checkpoint into a host Vec before model admission.
        // SAFETY: the read-only mapping owns every tensor view below.
        let safe = unsafe { candle_core::safetensors::MmapedSafetensors::new(&shard_path) }
            .map_err(|err| {
                Error::ModelLoadError(format!(
                    "Failed to parse Audio8 checkpoint {}: {err}",
                    shard_path.display()
                ))
            })?;
        let mut tensors = BTreeMap::new();
        let mut source_to_remapped = BTreeMap::new();
        for (source_name, view) in safe.tensors() {
            let remapped_name = remap_audio8_key(&source_name);
            if tensors.contains_key(&remapped_name) {
                return Err(Error::ModelLoadError(format!(
                    "Audio8 remapped tensor name collision: `{remapped_name}`"
                )));
            }
            source_to_remapped.insert(source_name.clone(), remapped_name.clone());
            tensors.insert(
                remapped_name.clone(),
                Audio8TensorSpec {
                    source_name,
                    remapped_name,
                    shard_file: shard_file.clone(),
                    shape: view.shape().to_vec(),
                    source_bytes: u64::try_from(view.data().len()).map_err(|_| {
                        Error::ModelLoadError("Audio8 source tensor size exceeds u64".into())
                    })?,
                },
            );
        }

        let resolved = Self {
            model_dir: model_dir.to_path_buf(),
            shard_files: vec![shard_file],
            tensors,
            source_to_remapped,
        };
        resolved.validate_required_contract_tensors()?;
        Ok(resolved)
    }

    pub fn shard_paths(&self) -> Vec<PathBuf> {
        self.shard_files
            .iter()
            .map(|file| self.model_dir.join(file))
            .collect()
    }

    pub fn shard_files(&self) -> &[String] {
        &self.shard_files
    }

    pub fn tensors(&self) -> &BTreeMap<String, Audio8TensorSpec> {
        &self.tensors
    }

    pub fn source_to_remapped(&self) -> &BTreeMap<String, String> {
        &self.source_to_remapped
    }

    pub fn tensor(&self, remapped_name: &str) -> Option<&Audio8TensorSpec> {
        self.tensors.get(remapped_name)
    }

    pub fn has_tensor(&self, remapped_name: &str) -> bool {
        self.tensors.contains_key(remapped_name)
    }

    pub(crate) fn memory_inventory(&self, dtype: DType) -> Result<Audio8TensorMemory> {
        let mut inventory = Audio8TensorMemory::default();
        for spec in self.tensors.values() {
            let elements = spec.shape.iter().try_fold(1_u64, |elements, dimension| {
                elements
                    .checked_mul(u64::try_from(*dimension).map_err(|_| {
                        Error::ModelLoadError("Audio8 tensor dimension exceeds u64".into())
                    })?)
                    .ok_or_else(|| Error::ModelLoadError("Audio8 tensor size overflow".into()))
            })?;
            let target_bytes = elements
                .checked_mul(dtype.size_in_bytes() as u64)
                .ok_or_else(|| Error::ModelLoadError("Audio8 tensor byte size overflow".into()))?;
            inventory.resident_bytes =
                checked_memory_sum(&[inventory.resident_bytes, target_bytes])?;
            inventory.source_bytes =
                checked_memory_sum(&[inventory.source_bytes, spec.source_bytes])?;
            inventory.largest_source_tensor_bytes =
                inventory.largest_source_tensor_bytes.max(spec.source_bytes);
            inventory.largest_target_tensor_bytes =
                inventory.largest_target_tensor_bytes.max(target_bytes);
        }
        Ok(inventory)
    }

    fn validate_required_contract_tensors(&self) -> Result<()> {
        for name in [
            "embed_tokens.weight",
            "norm.weight",
            "codebook_embeddings.weight",
            "fast_embeddings.weight",
            "fast_norm.weight",
            "fast_output.weight",
        ] {
            if !self.has_tensor(name) {
                return Err(Error::ModelLoadError(format!(
                    "Audio8 model weights missing required tensor `{name}`"
                )));
            }
        }
        Ok(())
    }
}

impl Audio8Weights {
    pub fn load(
        model_dir: &Path,
        device: DeviceProfile,
        dtype_override: Option<&str>,
    ) -> Result<Self> {
        let dtype = Self::resolve_dtype(model_dir, &device, dtype_override)?;
        Self::load_with_dtype(model_dir, device, dtype)
    }

    /// The dtype [`Self::load`] would pick: the override, else the device
    /// policy for the checkpoint's dtype.
    pub(crate) fn resolve_dtype(
        model_dir: &Path,
        device: &DeviceProfile,
        dtype_override: Option<&str>,
    ) -> Result<DType> {
        validate_audio8_backend(device)?;
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(&config_path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {err}", config_path.display()))
        })?;
        let checkpoint_dtype = checkpoint_dtype_from_config_json(&config_str);
        select_audio8_dtype(device, dtype_override, checkpoint_dtype)
    }

    /// Map the shards at `dtype` (already validated for `device`).
    pub(crate) fn load_with_dtype(
        model_dir: &Path,
        device: DeviceProfile,
        dtype: DType,
    ) -> Result<Self> {
        validate_audio8_backend(&device)?;
        let index = Audio8WeightIndex::load(model_dir)?;
        let shard_paths = index.shard_paths();
        let remapped_to_source = index
            .source_to_remapped()
            .iter()
            .map(|(source, remapped)| (remapped.clone(), source.clone()))
            .collect::<BTreeMap<_, _>>();
        // Host-cast, exact-size uploads with periodic synchronization on
        // Metal (`staged`): Candle's own reads cast every BF16 tensor on the
        // device into a power-of-two buffer (9.12 GB held in 12.15 GB at F16)
        // and kept every upload until the load fence.
        let vb = unsafe {
            crate::models::shared::weights::staged::metal_staged_safetensors(
                &shard_paths,
                dtype,
                &device.device,
            )?
        };
        let vb = vb.rename_f(move |name| {
            remapped_to_source
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        });
        Ok(Self {
            dtype,
            device: device.device,
            index,
            vb,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn index(&self) -> &Audio8WeightIndex {
        &self.index
    }

    pub fn var_builder(&self) -> VarBuilder<'static> {
        self.vb.clone()
    }
}

pub fn select_audio8_dtype(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
    checkpoint_dtype: Option<DType>,
) -> Result<DType> {
    let requested = dtype_override
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let dtype = match requested {
        Some(requested) => match requested.to_ascii_lowercase().as_str() {
            "f32" | "float32" | "torch.float32" => DType::F32,
            "f16" | "float16" | "torch.float16" => DType::F16,
            "bf16" | "bfloat16" | "torch.bfloat16" => DType::BF16,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "Invalid IZWI_AUDIO8_DTYPE {requested:?}: expected f32, f16, or bf16"
                )))
            }
        },
        None => match device.kind {
            DeviceKind::Cpu => DType::F32,
            DeviceKind::Metal => DType::F16,
            DeviceKind::Cuda => {
                let checkpoint = checkpoint_dtype.unwrap_or(DType::BF16);
                // Preserve BF16's exponent range when the selected CUDA device
                // cannot execute BF16. A lower-range F16 override remains explicit.
                if checkpoint == DType::BF16 && !audio8_cuda_bf16_supported(device) {
                    tracing::info!(
                        compute_capability = ?device.capabilities.cuda_compute_capability,
                        "Audio8 CUDA device cannot execute BF16; using F32 to preserve checkpoint range"
                    );
                    DType::F32
                } else {
                    checkpoint
                }
            }
        },
    };
    let supported = match (device.kind, dtype) {
        (_, DType::F32) => true,
        (DeviceKind::Metal, DType::F16) => device.capabilities.supports_f16,
        (DeviceKind::Cuda, DType::F16) => device.capabilities.supports_f16,
        (DeviceKind::Cuda, DType::BF16) => audio8_cuda_bf16_supported(device),
        _ => false,
    };
    if !supported {
        let reason = match device.kind {
            DeviceKind::Cpu => "Audio8 CPU execution requires F32",
            DeviceKind::Metal => "Audio8 Metal paged attention supports F16/F32, never BF16",
            DeviceKind::Cuda => "Audio8 CUDA BF16 requires observed compute capability 8.0 or newer; F16 requires device support",
        };
        return Err(Error::InvalidInput(format!(
            "Unsupported Audio8 dtype {dtype:?} on {:?}: {reason}",
            device.kind
        )));
    }
    Ok(dtype)
}

fn audio8_cuda_bf16_supported(device: &DeviceProfile) -> bool {
    device.capabilities.supports_bf16
        && device
            .capabilities
            .cuda_compute_capability
            .is_some_and(|(major, _)| major >= 8)
}

pub(crate) fn validate_audio8_backend(device: &DeviceProfile) -> Result<()> {
    let backend = BackendKind::from(device.kind);
    if !crate::backends::kv::managed_kv_backend_compiled(backend) {
        return Err(Error::ModelLoadError(format!(
            "Audio8 {backend:?} managed attention is not compiled into this binary"
        )));
    }
    let matches_device = match device.kind {
        DeviceKind::Cpu => device.device.is_cpu(),
        DeviceKind::Metal => device.device.is_metal(),
        DeviceKind::Cuda => device.device.is_cuda(),
    };
    if !matches_device {
        return Err(Error::ModelLoadError(format!(
            "Audio8 {backend:?} profile does not refer to an initialized {backend:?} device"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Audio8TensorMemory {
    pub(crate) resident_bytes: u64,
    pub(crate) source_bytes: u64,
    pub(crate) largest_source_tensor_bytes: u64,
    pub(crate) largest_target_tensor_bytes: u64,
}

/// Model tensors only. Invocation workspaces and managed KV have separate
/// authorities and must not be charged a second time to the weight lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Audio8ModelMemory {
    pub(crate) resident_bytes: u64,
    /// The load's peak in one memory (CPU and Metal) or on the device (CUDA).
    pub(crate) load_peak_bytes: u64,
    /// Metal: the part of `load_peak_bytes` the GPU allocates.
    pub(crate) metal_device_load_peak_bytes: u64,
    pub(crate) cuda_host_load_peak_bytes: u64,
}

pub(crate) fn audio8_model_memory(
    model_dir: &Path,
    device: &DeviceProfile,
) -> Result<Audio8ModelMemory> {
    validate_audio8_backend(device)?;
    let config_text = fs::read_to_string(model_dir.join("config.json"))?;
    let dtype_override = std::env::var(super::AUDIO8_DTYPE_ENV).ok();
    let dtype = select_audio8_dtype(
        device,
        dtype_override.as_deref(),
        checkpoint_dtype_from_config_json(&config_text),
    )?;
    let config: Audio8Config = serde_json::from_str(&config_text)?;
    config.validate()?;
    let index = Audio8WeightIndex::load(model_dir)?;
    let transformer = index.memory_inventory(dtype)?;
    let codec = PthTensorMap::open_first_non_empty(
        &model_dir.join("codec.pth"),
        &[None, Some("state_dict"), Some("generator")],
    )?;
    let codec = audio8_codec_memory(&codec.specs())?;
    let slow = Audio8SlowConfig::from_config(&config)?;
    let fast = Audio8FastConfig::from_config(&config)?;
    let auxiliary_bytes = checked_memory_sum(&[
        rotary_cache_bytes(slow.max_seq_len, slow.head_dim)?,
        rotary_cache_bytes(fast.num_codebooks, fast.head_dim)?,
        Audio8DacConfig::current().rotary_cache_bytes()?,
        // The compact output projection is gathered once; input embeddings stay full size.
        (config.codebook_size as u64 + 1)
            .checked_mul(config.text_config.hidden_size as u64)
            .and_then(|elements| elements.checked_mul(dtype.size_in_bytes() as u64))
            .ok_or_else(|| Error::ModelLoadError("Audio8 compact head size overflow".into()))?,
    ])?;
    audio8_memory_from_inventory(transformer, codec, auxiliary_bytes)
}

fn rotary_cache_bytes(positions: usize, head_dim: usize) -> Result<u64> {
    u64::try_from(positions)
        .ok()
        .and_then(|positions| positions.checked_mul(u64::try_from(head_dim).ok()?))
        .and_then(|elements| elements.checked_mul(DType::F32.size_in_bytes() as u64))
        .ok_or_else(|| Error::ModelLoadError("Audio8 rotary cache byte size overflow".into()))
}

fn audio8_memory_from_inventory(
    transformer: Audio8TensorMemory,
    codec: Audio8CodecMemory,
    auxiliary_bytes: u64,
) -> Result<Audio8ModelMemory> {
    let resident_bytes = checked_memory_sum(&[
        transformer.resident_bytes,
        codec.resident_parameter_bytes,
        auxiliary_bytes,
    ])?;
    // Candle safetensors conversion may temporarily retain both representations
    // of one tensor. The codec is read eagerly, then normalized; its raw map and
    // all fused weights coexist until DAC construction releases the builder.
    // During weight normalization, square/reduction/scale temporaries also live.
    let transformer_peak = checked_memory_sum(&[
        resident_bytes,
        transformer.largest_source_tensor_bytes,
        transformer.largest_target_tensor_bytes,
    ])?;
    let codec_peak = checked_memory_sum(&[
        transformer.resident_bytes,
        auxiliary_bytes,
        codec.raw_load_bytes,
        codec.fused_load_bytes,
        codec.largest_source_tensor_bytes,
        codec.largest_target_tensor_bytes,
        codec.largest_target_tensor_bytes,
        codec.largest_target_tensor_bytes,
    ])?;
    // Keep the checkpoint's read-only mappings in the host load envelope, plus
    // one safetensors staging copy if Candle needs aligned storage. PTH transfer
    // separately holds a raw byte buffer, a decoded source tensor, and its F32
    // conversion for one tensor at a time, never the whole codec.
    let codec_host_scratch_bytes = checked_memory_sum(&[
        codec.largest_source_tensor_bytes,
        codec.largest_source_tensor_bytes,
        codec.largest_target_tensor_bytes,
    ])?;
    let cuda_host_load_peak_bytes = checked_memory_sum(&[
        transformer.source_bytes,
        transformer
            .largest_source_tensor_bytes
            .max(codec_host_scratch_bytes),
        auxiliary_bytes,
    ])?;
    Ok(Audio8ModelMemory {
        resident_bytes,
        load_peak_bytes: transformer_peak.max(codec_peak),
        // The staged transformer uploads each tensor at its exact size after
        // casting it on the host, and the codec is normalized on the host
        // and uploaded tensor by tensor: on Metal the GPU holds only the
        // final tensors (measured: a 10.738 GB peak over a 10.737 GB
        // residency); the rest of `load_peak_bytes` is host memory.
        metal_device_load_peak_bytes: resident_bytes,
        cuda_host_load_peak_bytes,
    })
}

fn checked_memory_sum(bytes: &[u64]) -> Result<u64> {
    bytes.iter().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or_else(|| Error::ModelLoadError("Audio8 model memory estimate overflow".into()))
    })
}

pub fn audio8_vb_path(remapped_name: &str) -> String {
    remapped_name
        .strip_suffix(".weight")
        .or_else(|| remapped_name.strip_suffix(".bias"))
        .unwrap_or(remapped_name)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype as SafeDType, TensorView};
    use std::collections::HashMap;

    fn temp_model_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("izwi-audio8-weights-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    }

    fn write_shard(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        let buffers = tensors
            .iter()
            .map(|(name, shape, values)| ((*name).to_string(), shape.clone(), bytes(values)))
            .collect::<Vec<_>>();
        let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, data) in &buffers {
            let view = TensorView::new(SafeDType::F32, shape.clone(), data).unwrap();
            views.insert(name.clone(), view);
        }
        safetensors::serialize_to_file(&views, None, path).unwrap();
    }

    fn write_minimal_checkpoint(dir: &Path) {
        write_shard(
            &dir.join(AUDIO8_WEIGHTS_FILE),
            &[
                ("embeddings.weight", vec![4, 3], vec![0.0; 12]),
                ("norm.weight", vec![3], vec![1.0; 3]),
                ("codebook_embeddings.weight", vec![16, 3], vec![0.0; 48]),
                ("fast_embeddings.weight", vec![4, 5], vec![0.0; 20]),
                ("fast_norm.weight", vec![5], vec![1.0; 5]),
                ("fast_output.weight", vec![4, 5], vec![0.0; 20]),
                (
                    "fast_layers.0.attention.wqkv.weight",
                    vec![15, 5],
                    vec![0.0; 75],
                ),
            ],
        );
    }

    #[test]
    fn resolves_flat_checkpoint_with_remapped_names_and_shapes() {
        let dir = temp_model_dir();
        write_minimal_checkpoint(&dir);
        let index = Audio8WeightIndex::load(&dir).expect("index");
        assert_eq!(index.shard_files(), [AUDIO8_WEIGHTS_FILE]);
        assert_eq!(
            index
                .source_to_remapped()
                .get("fast_layers.0.attention.wqkv.weight")
                .unwrap(),
            "fast_layers.0.self_attn.qkv_proj.weight"
        );
        assert_eq!(
            index.tensor("embed_tokens.weight").unwrap().shape,
            vec![4, 3]
        );
        assert_eq!(
            index
                .tensor("fast_layers.0.self_attn.qkv_proj.weight")
                .unwrap()
                .shape,
            vec![15, 5]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn var_builder_loads_remapped_source_tensor_names() {
        let dir = temp_model_dir();
        write_minimal_checkpoint(&dir);
        std::fs::write(dir.join("config.json"), r#"{"dtype":"float32"}"#).unwrap();
        let weights = Audio8Weights::load(&dir, DeviceProfile::cpu(), None).expect("weights");
        let vb = weights.var_builder();
        assert!(vb.contains_tensor("embed_tokens.weight"));
        assert!(vb.contains_tensor("fast_layers.0.self_attn.qkv_proj.weight"));
        assert!(vb.get((4, 3), "embed_tokens.weight").is_ok());
        assert!(vb
            .get((15, 5), "fast_layers.0.self_attn.qkv_proj.weight")
            .is_ok());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_missing_required_contract_tensor() {
        let dir = temp_model_dir();
        write_shard(
            &dir.join(AUDIO8_WEIGHTS_FILE),
            &[
                ("embeddings.weight", vec![4, 3], vec![0.0; 12]),
                ("codebook_embeddings.weight", vec![16, 3], vec![0.0; 48]),
            ],
        );
        let err = Audio8WeightIndex::load(&dir).unwrap_err();
        assert!(err.to_string().contains("norm.weight"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dtype_policy_uses_f32_on_cpu_even_for_bf16_checkpoints() {
        let device = DeviceProfile::cpu();
        assert_eq!(
            select_audio8_dtype(&device, None, Some(DType::BF16)).unwrap(),
            DType::F32
        );
        assert!(select_audio8_dtype(&device, Some("bf16"), Some(DType::F32)).is_err());
        assert!(select_audio8_dtype(&device, Some("f16"), Some(DType::F32)).is_err());
    }

    #[test]
    fn dtype_policy_uses_f16_on_metal_to_avoid_f32_checkpoint_expansion() {
        let device = DeviceProfile {
            device: Device::Cpu,
            kind: crate::backends::DeviceKind::Metal,
            capabilities: crate::backends::DeviceCapabilities {
                supports_f16: true,
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };
        assert_eq!(
            select_audio8_dtype(&device, None, Some(DType::BF16)).unwrap(),
            DType::F16
        );
        assert_eq!(
            select_audio8_dtype(&device, Some("f32"), Some(DType::BF16)).unwrap(),
            DType::F32
        );
        let error = select_audio8_dtype(&device, Some("bf16"), Some(DType::BF16)).unwrap_err();
        assert!(error.to_string().contains("paged attention"));
    }

    fn synthetic_cuda_profile(compute_capability: Option<(u32, u32)>) -> DeviceProfile {
        DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: crate::backends::DeviceCapabilities {
                supports_f16: true,
                supports_bf16: compute_capability.is_some_and(|(major, _)| major >= 8),
                cuda_compute_capability: compute_capability,
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn cuda_dtype_policy_requires_observed_bf16_hardware_and_preserves_explicit_f16() {
        for compute_capability in [None, Some((7, 5))] {
            let device = synthetic_cuda_profile(compute_capability);
            assert_eq!(
                select_audio8_dtype(&device, None, Some(DType::BF16)).unwrap(),
                DType::F32
            );
            assert!(select_audio8_dtype(&device, Some("bf16"), Some(DType::BF16)).is_err());
            assert_eq!(
                select_audio8_dtype(&device, Some("f16"), Some(DType::BF16)).unwrap(),
                DType::F16
            );
        }
        let ampere = synthetic_cuda_profile(Some((8, 0)));
        assert_eq!(
            select_audio8_dtype(&ampere, None, Some(DType::BF16)).unwrap(),
            DType::BF16
        );
        let mut unsupported_f16 = synthetic_cuda_profile(Some((5, 2)));
        unsupported_f16.capabilities.supports_f16 = false;
        assert!(select_audio8_dtype(&unsupported_f16, Some("f16"), None).is_err());
    }

    #[test]
    fn dtype_override_rejects_typos_instead_of_silently_selecting_a_different_representation() {
        let error = select_audio8_dtype(&DeviceProfile::cpu(), Some("b16"), None).unwrap_err();
        assert!(error.to_string().contains("IZWI_AUDIO8_DTYPE"));
        assert!(error.to_string().contains("b16"));
    }

    #[test]
    fn synthetic_accelerator_profile_is_rejected_before_reading_artifacts() {
        let directory = std::env::temp_dir().join("fish-s2-artifacts-must-not-be-read");
        let error =
            match Audio8Weights::load(&directory, synthetic_cuda_profile(Some((8, 0))), None) {
                Ok(_) => panic!("CPU storage must not pass CUDA preflight"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("CUDA") || error.to_string().contains("Cuda"));
        assert!(!error.to_string().contains("config.json"));
    }

    #[test]
    fn indexed_inventory_counts_target_dtype_without_allocating_weight_tensors() {
        let dir = temp_model_dir();
        write_minimal_checkpoint(&dir);
        let index = Audio8WeightIndex::load(&dir).unwrap();
        let f32 = index.memory_inventory(DType::F32).unwrap();
        let f16 = index.memory_inventory(DType::F16).unwrap();
        assert_eq!(f32.source_bytes, 183 * 4);
        assert_eq!(f32.resident_bytes, 183 * 4);
        assert_eq!(f16.resident_bytes, 183 * 2);
        assert_eq!(f16.source_bytes, f32.source_bytes);
        assert_eq!(f16.largest_target_tensor_bytes, 75 * 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn audio8_memory_prices_bf16_expansion_codec_overlap_and_shared_rope() {
        // Header-only inventory for the pinned checkpoint, not allocated tensors.
        const TRANSFORMER_ELEMENTS: u64 = 4_561_852_416;
        const LARGEST_SOURCE_TENSOR: u64 = 797_573_120;
        let codec = Audio8CodecMemory {
            resident_parameter_bytes: 1_565_593_540,
            // The F32 weights; the 6,291,456 bytes of `freqs_cis` are not read.
            raw_load_bytes: 1_565_722_120,
            fused_load_bytes: 306_057_600,
            largest_source_tensor_bytes: 75_497_472,
            largest_target_tensor_bytes: 75_497_472,
        };
        let auxiliary_bytes = checked_memory_sum(&[
            rotary_cache_bytes(32_768, 128).unwrap(),
            rotary_cache_bytes(10, 128).unwrap(),
            6_291_456,
        ])
        .unwrap();
        let estimate = |bytes_per_element| {
            audio8_memory_from_inventory(
                Audio8TensorMemory {
                    resident_bytes: TRANSFORMER_ELEMENTS * bytes_per_element,
                    source_bytes: TRANSFORMER_ELEMENTS * 2,
                    largest_source_tensor_bytes: LARGEST_SOURCE_TENSOR,
                    largest_target_tensor_bytes: LARGEST_SOURCE_TENSOR / 2 * bytes_per_element,
                },
                codec,
                auxiliary_bytes,
            )
            .unwrap()
        };
        let bf16 = estimate(2);
        let f32 = estimate(4);
        assert_eq!(bf16.resident_bytes, 10_712_372_164);
        assert_eq!(bf16.load_peak_bytes, 12_307_518_404);
        assert_eq!(bf16.cuda_host_load_peak_bytes, 9_944_351_744);
        assert_eq!(f32.load_peak_bytes, 22_228_796_356);
        assert_eq!(
            f32.resident_bytes - bf16.resident_bytes,
            TRANSFORMER_ELEMENTS * 2
        );
        assert!(bf16.load_peak_bytes > bf16.resident_bytes);
        assert!(bf16.load_peak_bytes < 24 * 1024 * 1024 * 1024);
        assert_eq!(
            bf16.cuda_host_load_peak_bytes,
            f32.cuda_host_load_peak_bytes
        );
        assert!(audio8_memory_from_inventory(
            Audio8TensorMemory {
                resident_bytes: u64::MAX,
                ..Default::default()
            },
            codec,
            auxiliary_bytes,
        )
        .is_err());
    }
}

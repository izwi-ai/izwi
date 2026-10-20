//! Fish S2 codec artifact boundary.
//!
//! The public S2 Pro checkpoint ships `codec.pth` as a PyTorch state dict. This
//! module owns the Fish-specific state-dict normalization before the actual DAC
//! encoder/decoder modules consume ordinary Candle tensors.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::{Error, Result};
use crate::models::shared::weights::pytorch::{PthTensorMap, PthTensorSpec};

#[path = "workspace.rs"]
mod workspace;
#[cfg(test)]
use workspace::maximum_decode_workspace_bytes;
pub(crate) use workspace::{
    decode_workspace_bytes, fft_workspace, maximum_preparation_workspace_bytes,
    preparation_workspace_bytes,
};

/// Conservative transient workspace for exact stateful decoding. Extend the
/// stateless envelope by the largest causal support: the post-transformer KV
/// window (127 prior frames) dominates every decoder convolution's history
/// (at most 54 samples at an already-upsampled resolution). Retained histories
/// and their rollback checkpoint are charged separately by the request owner.
/// This intentionally overprices downstream convolutions until per-stage CUDA
/// high-water measurements justify a tighter history-aware envelope.
pub(crate) fn streaming_decode_workspace_bytes(frames: usize) -> Result<u64> {
    use super::dac::FishS2DacConfig;
    if frames == 0 || frames > FishS2DacConfig::MAX_QUANTIZER_FRAMES {
        return decode_workspace_bytes(frames);
    }
    let history = FishS2DacConfig::current()
        .transformer_window_size
        .saturating_sub(1)
        .max(54);
    decode_workspace_bytes(
        frames
            .saturating_add(history)
            .min(FishS2DacConfig::MAX_QUANTIZER_FRAMES),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishS2CodecSupport {
    NativePthStateDict,
}

#[derive(Debug, Clone)]
pub struct FishS2CodecArtifact {
    pub path: PathBuf,
    pub support: FishS2CodecSupport,
}

#[derive(Debug)]
pub struct FishS2CodecWeights {
    tensors: HashMap<String, Tensor>,
    specs: Vec<PthTensorSpec>,
    source_key: Option<String>,
    device: Device,
    dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FishS2CodecMemory {
    pub(crate) resident_parameter_bytes: u64,
    pub(crate) raw_load_bytes: u64,
    pub(crate) fused_load_bytes: u64,
    pub(crate) largest_source_tensor_bytes: u64,
    pub(crate) largest_target_tensor_bytes: u64,
}

/// Prices the current F32 codec loader from tensor metadata, without reading payloads.
pub(crate) fn fish_s2_codec_memory(specs: &[PthTensorSpec]) -> Result<FishS2CodecMemory> {
    let mut memory = FishS2CodecMemory {
        resident_parameter_bytes: 0,
        raw_load_bytes: 0,
        fused_load_bytes: 0,
        largest_source_tensor_bytes: 0,
        largest_target_tensor_bytes: 0,
    };
    let has_generator = specs.iter().any(|spec| spec.name.contains("generator."));
    let mut normalized = BTreeMap::new();
    for spec in specs {
        if is_fish_s2_codec_buffer(&spec.name) {
            // Never read (`FishS2CodecWeights::load`).
            continue;
        }
        let source_bytes = codec_spec_bytes(spec, spec.dtype)?;
        let target_bytes = codec_spec_bytes(spec, DType::F32)?;
        // The host read precedes generator selection and normalization in the loader.
        memory.raw_load_bytes = checked_codec_bytes(memory.raw_load_bytes, target_bytes)?;
        memory.largest_source_tensor_bytes = memory.largest_source_tensor_bytes.max(source_bytes);
        memory.largest_target_tensor_bytes = memory.largest_target_tensor_bytes.max(target_bytes);
        if !has_generator || spec.name.contains("generator.") {
            normalized.insert(normalize_codec_key(&spec.name, has_generator), spec);
        }
    }

    let mut weight_norm =
        BTreeMap::<String, (Option<&PthTensorSpec>, Option<&PthTensorSpec>)>::new();
    for (name, spec) in &normalized {
        if name.ends_with(".freqs_cis") || name.ends_with(".causal_mask") {
            // The native codec regenerates its immutable positional state.
            continue;
        }
        if let Some(base) = name.strip_suffix(".parametrizations.weight.original0") {
            weight_norm.entry(base.to_owned()).or_default().0 = Some(spec);
        } else if let Some(base) = name.strip_suffix(".parametrizations.weight.original1") {
            weight_norm.entry(base.to_owned()).or_default().1 = Some(spec);
        } else if let Some(base) = name.strip_suffix(".weight_g") {
            weight_norm
                .entry(base.to_owned())
                .or_default()
                .0
                .get_or_insert(spec);
        } else if let Some(base) = name.strip_suffix(".weight_v") {
            weight_norm
                .entry(base.to_owned())
                .or_default()
                .1
                .get_or_insert(spec);
        } else {
            memory.resident_parameter_bytes = checked_codec_bytes(
                memory.resident_parameter_bytes,
                codec_spec_bytes(spec, DType::F32)?,
            )?;
        }
    }
    for (base, (weight_g, weight_v)) in weight_norm {
        if normalized.contains_key(&format!("{base}.weight")) {
            continue;
        }
        let (Some(_), Some(weight_v)) = (weight_g, weight_v) else {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 codec has an incomplete weight_norm pair at {base}"
            )));
        };
        let bytes = codec_spec_bytes(weight_v, DType::F32)?;
        memory.resident_parameter_bytes =
            checked_codec_bytes(memory.resident_parameter_bytes, bytes)?;
        memory.fused_load_bytes = checked_codec_bytes(memory.fused_load_bytes, bytes)?;
    }
    Ok(memory)
}

fn codec_spec_bytes(spec: &PthTensorSpec, dtype: DType) -> Result<u64> {
    spec.shape
        .iter()
        .try_fold(dtype.size_in_bytes() as u64, |bytes, dim| {
            bytes.checked_mul(u64::try_from(*dim).ok()?)
        })
        .ok_or_else(|| Error::ModelLoadError("Fish S2 codec tensor byte size overflowed".into()))
}

fn checked_codec_bytes(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::ModelLoadError("Fish S2 codec byte size overflowed".into()))
}

impl FishS2CodecArtifact {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("codec.pth");
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 codec artifact missing: {}",
                path.display()
            )));
        }
        Ok(Self {
            path,
            support: FishS2CodecSupport::NativePthStateDict,
        })
    }

    pub fn ensure_native_supported(&self) -> Result<()> {
        match self.support {
            FishS2CodecSupport::NativePthStateDict => Ok(()),
        }
    }

    pub fn load_weights(&self, device: &Device, dtype: DType) -> Result<FishS2CodecWeights> {
        self.ensure_native_supported()?;
        FishS2CodecWeights::load(&self.path, device, dtype)
    }
}

impl FishS2CodecWeights {
    /// Read the codec on the host, without the positional buffers the DAC
    /// regenerates, and fuse its weight-norm pairs there; then upload what
    /// the DAC reads, one tensor at a time, contiguous and at its exact size.
    /// Read and fused on the device, the codec kept every weight-norm pair
    /// beside its fused weight until the load fence, and each fused weight in
    /// a power-of-two buffer for the model's life.
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let archive = PthTensorMap::open_first_non_empty(
            path,
            &[None, Some("state_dict"), Some("generator")],
        )?;
        let specs = archive.specs();
        let raw = archive.read_where(&Device::Cpu, Some(dtype), |name| {
            !is_fish_s2_codec_buffer(name)
        })?;
        let host = normalize_fish_s2_codec_state_dict(raw)?;
        validate_codec_state_dict(&host)?;
        let tensors = upload_codec_weights(host, device)?;
        Ok(Self {
            tensors,
            specs,
            source_key: archive.selected_key().map(str::to_string),
            device: device.clone(),
            dtype,
        })
    }

    #[cfg(test)]
    fn from_raw_tensors(
        raw: BTreeMap<String, Tensor>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let specs = raw
            .iter()
            .map(|(name, tensor)| PthTensorSpec {
                name: name.clone(),
                dtype: tensor.dtype(),
                shape: tensor.dims().to_vec(),
                archive_member_path: format!("synthetic/{name}"),
            })
            .collect();
        let tensors = normalize_fish_s2_codec_state_dict(raw)?;
        validate_codec_state_dict(&tensors)?;
        Ok(Self {
            tensors,
            specs,
            source_key: None,
            device,
            dtype,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn source_tensor_count(&self) -> usize {
        self.specs.len()
    }

    pub fn source_key(&self) -> Option<&str> {
        self.source_key.as_deref()
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn var_builder(&self) -> VarBuilder<'_> {
        VarBuilder::from_tensors(self.tensors.clone(), self.dtype, &self.device)
    }
}

/// The codec's positional buffers (`freqs_cis`, `causal_mask`), which the
/// native DAC regenerates: never read or priced. (Candle 0.11 already skips
/// the boolean masks, 1.21 GB at F32, which it cannot parse.)
pub(crate) fn is_fish_s2_codec_buffer(name: &str) -> bool {
    name.ends_with(".freqs_cis") || name.ends_with(".causal_mask")
}

/// The normalized host codec on `device`: every tensor the DAC reads,
/// contiguous, at its exact size; a weight-norm pair whose fused `weight` is
/// present stays behind. Each host tensor is dropped once uploaded.
fn upload_codec_weights(
    host: HashMap<String, Tensor>,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    const PAIR_SUFFIXES: [&str; 4] = [
        ".weight_g",
        ".weight_v",
        ".parametrizations.weight.original0",
        ".parametrizations.weight.original1",
    ];
    let fused = |name: &str| {
        PAIR_SUFFIXES.iter().any(|suffix| {
            name.strip_suffix(suffix)
                .is_some_and(|base| host.contains_key(&format!("{base}.weight")))
        })
    };
    let names = host
        .keys()
        .filter(|name| !fused(name))
        .cloned()
        .collect::<Vec<_>>();
    let mut host = host;
    let mut tensors = HashMap::with_capacity(names.len());
    for name in names {
        let tensor = host.remove(&name).expect("listed codec tensor");
        tensors.insert(name, tensor.contiguous()?.to_device(device)?);
    }
    Ok(tensors)
}

fn normalize_fish_s2_codec_state_dict(
    raw: BTreeMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut filtered = BTreeMap::new();
    let has_generator = raw.keys().any(|key| key.contains("generator."));
    for (key, tensor) in raw {
        if has_generator && !key.contains("generator.") {
            continue;
        }
        let normalized = normalize_codec_key(&key, has_generator);
        filtered.insert(normalized, tensor);
    }

    let mut tensors = HashMap::new();
    let mut weight_norm_pairs = BTreeMap::<String, (Option<Tensor>, Option<Tensor>)>::new();
    for (key, tensor) in filtered {
        if let Some(base) = key.strip_suffix(".parametrizations.weight.original0") {
            let base = base.to_string();
            tensors.insert(format!("{base}.weight_g"), tensor.clone());
            weight_norm_pairs.entry(base).or_default().0 = Some(tensor.clone());
        } else if let Some(base) = key.strip_suffix(".parametrizations.weight.original1") {
            let base = base.to_string();
            tensors.insert(format!("{base}.weight_v"), tensor.clone());
            weight_norm_pairs.entry(base).or_default().1 = Some(tensor.clone());
        }
        tensors.insert(key, tensor);
    }

    for (base, (weight_g, weight_v)) in weight_norm_pairs {
        if tensors.contains_key(&format!("{base}.weight")) {
            continue;
        }
        if let (Some(weight_g), Some(weight_v)) = (weight_g, weight_v) {
            let fused = fuse_weight_norm_dim0(&weight_v, &weight_g)?;
            tensors.insert(format!("{base}.weight"), fused);
        }
    }

    Ok(tensors)
}

fn normalize_codec_key(key: &str, generator_only: bool) -> String {
    let mut normalized = key.to_string();
    for prefix in ["state_dict.", "module.", "model.", "codec."] {
        if let Some(stripped) = normalized.strip_prefix(prefix) {
            normalized = stripped.to_string();
        }
    }
    if generator_only {
        normalized = normalized.replace("generator.", "");
    } else if let Some(stripped) = normalized.strip_prefix("generator.") {
        normalized = stripped.to_string();
    }
    normalized
}

pub(crate) fn fuse_weight_norm_dim0(weight_v: &Tensor, weight_g: &Tensor) -> Result<Tensor> {
    let rank = weight_v.rank();
    let sq = weight_v.sqr()?;
    let norm = match rank {
        2 => sq.sum_keepdim(1)?,
        3 => sq.sum_keepdim((1, 2))?,
        _ => {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 codec cannot fuse weight_norm tensor rank {rank}"
            )))
        }
    }
    .sqrt()?;
    let scale = weight_g.broadcast_div(&norm)?;
    weight_v.broadcast_mul(&scale).map_err(Error::from)
}

fn validate_codec_state_dict(tensors: &HashMap<String, Tensor>) -> Result<()> {
    let required = [
        ("encoder.", "encoder"),
        ("decoder.", "decoder"),
        ("quantizer.semantic_quantizer.", "semantic quantizer"),
        ("quantizer.quantizer.", "residual quantizer"),
    ];
    for (prefix, label) in required {
        if !tensors.keys().any(|name| name.starts_with(prefix)) {
            return Err(Error::ModelLoadError(format!(
                "Fish S2 codec.pth missing {label} tensors with prefix `{prefix}`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn spec(name: &str, shape: &[usize], dtype: DType) -> PthTensorSpec {
        PthTensorSpec {
            name: name.to_owned(),
            shape: shape.to_vec(),
            dtype,
            archive_member_path: format!("synthetic/{name}"),
        }
    }

    #[test]
    fn streaming_workspace_covers_history_and_rejects_invalid_quanta() {
        assert_eq!(
            streaming_decode_workspace_bytes(16).unwrap(),
            decode_workspace_bytes(143).unwrap()
        );
        assert!(
            streaming_decode_workspace_bytes(16).unwrap() > decode_workspace_bytes(16).unwrap()
        );
        assert_eq!(
            streaming_decode_workspace_bytes(4096).unwrap(),
            maximum_decode_workspace_bytes().unwrap()
        );
        assert!(streaming_decode_workspace_bytes(0).is_err());
        assert!(streaming_decode_workspace_bytes(4097).is_err());
    }

    #[test]
    fn codec_memory_prices_selected_fused_weights_and_eager_raw_load() {
        let specs = [
            spec(
                "generator.encoder.conv.parametrizations.weight.original0",
                &[2, 1, 1],
                DType::F32,
            ),
            spec(
                "generator.encoder.conv.parametrizations.weight.original1",
                &[2, 3, 1],
                DType::F16,
            ),
            spec("generator.encoder.conv.bias", &[2], DType::F32),
            spec("generator.encoder.freqs_cis", &[2, 2, 2], DType::BF16),
            spec("discriminator.weight", &[4_096], DType::F32),
            spec(
                "generator.quantizer.in_proj.weight_g",
                &[2, 1, 1],
                DType::F32,
            ),
            spec(
                "generator.quantizer.in_proj.weight_v",
                &[2, 2, 1],
                DType::F32,
            ),
        ];
        assert_eq!(
            fish_s2_codec_memory(&specs).unwrap(),
            FishS2CodecMemory {
                resident_parameter_bytes: 48,
                // `freqs_cis` is never read.
                raw_load_bytes: 16_448,
                fused_load_bytes: 40,
                largest_source_tensor_bytes: 16_384,
                largest_target_tensor_bytes: 16_384,
            }
        );
    }

    #[test]
    fn codec_memory_does_not_duplicate_existing_fused_weights() {
        let specs = [
            spec("encoder.conv.weight", &[2, 3, 1], DType::F32),
            spec("encoder.conv.weight_g", &[2, 1, 1], DType::F32),
            spec("encoder.conv.weight_v", &[2, 3, 1], DType::F32),
        ];
        let memory = fish_s2_codec_memory(&specs).unwrap();
        assert_eq!(memory.resident_parameter_bytes, 24);
        assert_eq!(memory.raw_load_bytes, 56);
        assert_eq!(memory.fused_load_bytes, 0);
        assert!(fish_s2_codec_memory(&[spec("encoder.conv.weight_g", &[2], DType::F32)]).is_err());
        assert!(
            fish_s2_codec_memory(&[spec("encoder.weight", &[usize::MAX], DType::F32)]).is_err()
        );
    }

    #[test]
    fn codec_artifact_reports_native_pth_loader() {
        let dir = std::env::temp_dir().join(format!("izwi-fish-s2-codec-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("codec.pth"), [0u8]).unwrap();

        let codec = FishS2CodecArtifact::load(&dir).unwrap();
        assert_eq!(codec.support, FishS2CodecSupport::NativePthStateDict);
        codec.ensure_native_supported().unwrap();

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn normalizes_generator_state_dict_and_weight_norm_aliases() {
        let device = Device::Cpu;
        let mut raw = BTreeMap::new();
        raw.insert(
            "generator.encoder.block.0.conv.parametrizations.weight.original0".to_string(),
            Tensor::from_vec(vec![2.0f32, 4.0], (2, 1, 1), &device).unwrap(),
        );
        raw.insert(
            "generator.encoder.block.0.conv.parametrizations.weight.original1".to_string(),
            Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0, 6.0, 8.0], (2, 1, 3), &device).unwrap(),
        );
        raw.insert(
            "generator.encoder.block.0.conv.bias".to_string(),
            Tensor::from_vec(vec![0.0f32, 0.0], (2,), &device).unwrap(),
        );
        raw.insert(
            "generator.decoder.model.0.conv.weight".to_string(),
            Tensor::zeros((1, 2, 3), candle_core::DType::F32, &device).unwrap(),
        );
        raw.insert(
            "generator.quantizer.semantic_quantizer.quantizers.0.codebook.weight".to_string(),
            Tensor::zeros((4096, 8), candle_core::DType::F32, &device).unwrap(),
        );
        raw.insert(
            "generator.quantizer.quantizer.quantizers.0.codebook.weight".to_string(),
            Tensor::zeros((1024, 8), candle_core::DType::F32, &device).unwrap(),
        );
        raw.insert(
            "discriminator.conv.weight".to_string(),
            Tensor::zeros((1, 1, 1), candle_core::DType::F32, &device).unwrap(),
        );

        let weights =
            FishS2CodecWeights::from_raw_tensors(raw, device, candle_core::DType::F32).unwrap();
        assert!(weights.contains_tensor("encoder.block.0.conv.weight_g"));
        assert!(weights.contains_tensor("encoder.block.0.conv.weight_v"));
        assert!(weights.contains_tensor("encoder.block.0.conv.weight"));
        assert!(!weights.contains_tensor("discriminator.conv.weight"));

        let fused = weights
            .tensor("encoder.block.0.conv.weight")
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((fused[0][0][0] - 1.2).abs() < 1e-5);
        assert!((fused[0][0][1] - 1.6).abs() < 1e-5);
        assert!((fused[1][0][1] - 2.4).abs() < 1e-5);
        assert!((fused[1][0][2] - 3.2).abs() < 1e-5);
    }

    #[test]
    fn rejects_codec_state_dict_without_residual_quantizer() {
        let device = Device::Cpu;
        let mut raw = BTreeMap::new();
        raw.insert(
            "encoder.block.0.conv.weight".to_string(),
            Tensor::zeros((1, 1, 1), candle_core::DType::F32, &device).unwrap(),
        );
        raw.insert(
            "decoder.model.0.conv.weight".to_string(),
            Tensor::zeros((1, 1, 1), candle_core::DType::F32, &device).unwrap(),
        );
        raw.insert(
            "quantizer.semantic_quantizer.quantizers.0.codebook.weight".to_string(),
            Tensor::zeros((4096, 8), candle_core::DType::F32, &device).unwrap(),
        );

        let err =
            FishS2CodecWeights::from_raw_tensors(raw, device, candle_core::DType::F32).unwrap_err();
        assert!(err.to_string().contains("residual quantizer"));
    }
}

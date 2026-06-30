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
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let archive = PthTensorMap::open_first_non_empty(
            path,
            &[None, Some("state_dict"), Some("generator")],
        )?;
        let specs = archive.specs();
        let raw = archive.read_all(device, Some(dtype))?;
        let tensors = normalize_fish_s2_codec_state_dict(raw)?;
        validate_codec_state_dict(&tensors)?;
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

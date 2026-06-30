//! Fish S2 sharded safetensor loading.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::contracts::remap_fish_qwen3_omni_key;
use crate::models::shared::config::checkpoint_dtype_from_config_json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2TensorSpec {
    pub source_name: String,
    pub remapped_name: String,
    pub shard_file: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FishS2WeightIndex {
    model_dir: PathBuf,
    shard_files: Vec<String>,
    tensors: BTreeMap<String, FishS2TensorSpec>,
    source_to_remapped: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct FishS2Weights {
    dtype: DType,
    device: Device,
    index: FishS2WeightIndex,
    vb: VarBuilder<'static>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

impl FishS2WeightIndex {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let raw = fs::read_to_string(&index_path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {err}", index_path.display()))
        })?;
        let index: SafetensorsIndex = serde_json::from_str(&raw).map_err(|err| {
            Error::ModelLoadError(format!("Failed to parse {}: {err}", index_path.display()))
        })?;

        let mut shard_files = BTreeSet::new();
        for shard_file in index.weight_map.values() {
            validate_shard_name(shard_file)?;
            shard_files.insert(shard_file.clone());
        }
        if shard_files.is_empty() {
            return Err(Error::ModelLoadError(
                "Fish S2 safetensors index contains no shard files".to_string(),
            ));
        }

        let mut tensors = BTreeMap::new();
        let mut source_to_remapped = BTreeMap::new();
        for shard_file in &shard_files {
            let shard_path = model_dir.join(shard_file);
            let data = fs::read(&shard_path).map_err(|err| {
                Error::ModelLoadError(format!(
                    "Failed to read Fish S2 safetensors shard {}: {err}",
                    shard_path.display()
                ))
            })?;
            let safe = SafeTensors::deserialize(&data).map_err(|err| {
                Error::ModelLoadError(format!(
                    "Failed to parse Fish S2 safetensors shard {}: {err}",
                    shard_path.display()
                ))
            })?;
            for (source_name, indexed_shard) in &index.weight_map {
                if indexed_shard != shard_file {
                    continue;
                }
                let view = safe.tensor(source_name).map_err(|err| {
                    Error::ModelLoadError(format!(
                        "Fish S2 index tensor `{source_name}` missing from shard {shard_file}: {err}"
                    ))
                })?;
                let remapped_name = remap_fish_qwen3_omni_key(source_name);
                if tensors.contains_key(&remapped_name) {
                    return Err(Error::ModelLoadError(format!(
                        "Fish S2 remapped tensor name collision: `{remapped_name}`"
                    )));
                }
                source_to_remapped.insert(source_name.clone(), remapped_name.clone());
                tensors.insert(
                    remapped_name.clone(),
                    FishS2TensorSpec {
                        source_name: source_name.clone(),
                        remapped_name,
                        shard_file: shard_file.clone(),
                        shape: view.shape().to_vec(),
                    },
                );
            }
        }

        let resolved = Self {
            model_dir: model_dir.to_path_buf(),
            shard_files: shard_files.into_iter().collect(),
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

    pub fn tensors(&self) -> &BTreeMap<String, FishS2TensorSpec> {
        &self.tensors
    }

    pub fn source_to_remapped(&self) -> &BTreeMap<String, String> {
        &self.source_to_remapped
    }

    pub fn tensor(&self, remapped_name: &str) -> Option<&FishS2TensorSpec> {
        self.tensors.get(remapped_name)
    }

    pub fn has_tensor(&self, remapped_name: &str) -> bool {
        self.tensors.contains_key(remapped_name)
    }

    fn validate_required_contract_tensors(&self) -> Result<()> {
        for name in [
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
            "codebook_embeddings.weight",
            "fast_embeddings.weight",
            "fast_norm.weight",
            "fast_output.weight",
        ] {
            if !self.has_tensor(name) {
                return Err(Error::ModelLoadError(format!(
                    "Fish S2 model weights missing required tensor `{name}`"
                )));
            }
        }
        Ok(())
    }
}

impl FishS2Weights {
    pub fn load(
        model_dir: &Path,
        device: DeviceProfile,
        dtype_override: Option<&str>,
    ) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(&config_path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {err}", config_path.display()))
        })?;
        let checkpoint_dtype = checkpoint_dtype_from_config_json(&config_str);
        let dtype = select_fish_s2_dtype(&device, dtype_override, checkpoint_dtype);
        let index = FishS2WeightIndex::load(model_dir)?;
        let shard_paths = index.shard_paths();
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)? };
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

    pub fn index(&self) -> &FishS2WeightIndex {
        &self.index
    }

    pub fn var_builder(&self) -> VarBuilder<'static> {
        self.vb.clone()
    }
}

pub fn select_fish_s2_dtype(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
    checkpoint_dtype: Option<DType>,
) -> DType {
    if let Some(requested) = dtype_override {
        match requested.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "torch.float32" => return DType::F32,
            "f16" | "float16" | "torch.float16" => return DType::F16,
            "bf16" | "bfloat16" | "torch.bfloat16" => return DType::BF16,
            _ => {}
        }
    }

    if device.kind.is_cpu() || device.kind.is_metal() {
        return DType::F32;
    }

    checkpoint_dtype.unwrap_or(DType::BF16)
}

pub fn fish_s2_vb_path(remapped_name: &str) -> String {
    remapped_name
        .strip_suffix(".weight")
        .or_else(|| remapped_name.strip_suffix(".bias"))
        .unwrap_or(remapped_name)
        .to_string()
}

fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path.components().count() != 1 || name.contains("..") || name.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Fish S2 safetensors index contains unsafe shard path `{name}`"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype as SafeDType, TensorView};
    use std::collections::HashMap;

    fn temp_model_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("izwi-fish-s2-weights-{}", uuid::Uuid::new_v4()));
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
        safetensors::serialize_to_file(&views, &None, path).unwrap();
    }

    fn write_minimal_index(dir: &Path) {
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{
                "text_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
                "text_model.model.norm.weight":"model-00001-of-00002.safetensors",
                "text_model.lm_head.weight":"model-00001-of-00002.safetensors",
                "audio_decoder.codebook_embeddings.weight":"model-00002-of-00002.safetensors",
                "audio_decoder.embeddings.weight":"model-00002-of-00002.safetensors",
                "audio_decoder.norm.weight":"model-00002-of-00002.safetensors",
                "audio_decoder.output.weight":"model-00002-of-00002.safetensors",
                "audio_decoder.layers.0.self_attn.qkv_proj.weight":"model-00002-of-00002.safetensors"
            }}"#,
        )
        .unwrap();
        write_shard(
            &dir.join("model-00001-of-00002.safetensors"),
            &[
                (
                    "text_model.model.embed_tokens.weight",
                    vec![4, 3],
                    vec![0.0; 12],
                ),
                ("text_model.model.norm.weight", vec![3], vec![1.0; 3]),
                ("text_model.lm_head.weight", vec![4, 3], vec![0.0; 12]),
            ],
        );
        write_shard(
            &dir.join("model-00002-of-00002.safetensors"),
            &[
                (
                    "audio_decoder.codebook_embeddings.weight",
                    vec![16, 3],
                    vec![0.0; 48],
                ),
                ("audio_decoder.embeddings.weight", vec![4, 5], vec![0.0; 20]),
                ("audio_decoder.norm.weight", vec![5], vec![1.0; 5]),
                ("audio_decoder.output.weight", vec![4, 5], vec![0.0; 20]),
                (
                    "audio_decoder.layers.0.self_attn.qkv_proj.weight",
                    vec![15, 5],
                    vec![0.0; 75],
                ),
            ],
        );
    }

    #[test]
    fn resolves_sharded_weight_index_with_remapped_names_and_shapes() {
        let dir = temp_model_dir();
        write_minimal_index(&dir);
        let index = FishS2WeightIndex::load(&dir).expect("index");
        assert_eq!(index.shard_files().len(), 2);
        assert_eq!(
            index
                .source_to_remapped()
                .get("audio_decoder.layers.0.self_attn.qkv_proj.weight")
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
    fn rejects_missing_required_contract_tensor() {
        let dir = temp_model_dir();
        write_minimal_index(&dir);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{
                "text_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
                "audio_decoder.codebook_embeddings.weight":"model-00002-of-00002.safetensors"
            }}"#,
        )
        .unwrap();
        let err = FishS2WeightIndex::load(&dir).unwrap_err();
        assert!(err.to_string().contains("norm.weight"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dtype_policy_uses_f32_on_cpu_even_for_bf16_checkpoints() {
        let device = DeviceProfile::cpu();
        assert_eq!(
            select_fish_s2_dtype(&device, None, Some(DType::BF16)),
            DType::F32
        );
        assert_eq!(
            select_fish_s2_dtype(&device, Some("bf16"), Some(DType::F32)),
            DType::BF16
        );
    }
}

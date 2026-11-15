//! The sharded checkpoint: index, memory maps and census.
//!
//! `model.safetensors.index.json` maps every tensor to one of the 16
//! shards. The census derives the exact tensor set (name, dtype, shape) from
//! the config and fails the load on any missing, extra or reshaped tensor
//! before a byte is uploaded. Weights are BF16 except the three router
//! tensors of each layer, which are F32.
//!
//! `lm_head.weight` is stored although the head is tied to the embedding.
//! The census requires it (so a checkpoint that untied it is noticed by
//! [`SovCheckpoint::head_is_tied`]) and the loader never uploads it.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;

use super::config::{SovLayerKind, SovTranscribeConfig};
use crate::error::{Error, Result};

pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const ENCODER_PREFIX: &str = "model.audio_tower";
pub const PROJECTOR_PREFIX: &str = "model.embed_audio";
pub const DECODER_PREFIX: &str = "model.language_model";
pub const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
pub const LM_HEAD: &str = "lm_head.weight";

/// The shard names of an `n`-shard checkpoint.
pub fn shard_files(shards: usize) -> Vec<String> {
    (1..=shards)
        .map(|shard| format!("model-{shard:05}-of-{shards:05}.safetensors"))
        .collect()
}

/// Shards of the published checkpoint.
pub const PUBLISHED_SHARDS: usize = 16;

/// One tensor the census expects.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CensusEntry {
    pub name: String,
    pub f32: bool,
    pub shape: Vec<usize>,
}

fn entry(name: impl Into<String>, shape: &[usize]) -> CensusEntry {
    CensusEntry {
        name: name.into(),
        f32: false,
        shape: shape.to_vec(),
    }
}

/// Every tensor of a checkpoint with `config`'s geometry.
pub fn expected_census(config: &SovTranscribeConfig) -> Vec<CensusEntry> {
    let e = &config.encoder;
    let d = &config.decoder;
    let mut out = Vec::new();
    let a = ENCODER_PREFIX;
    out.push(entry(format!("{a}.conv1.weight"), &[e.width, e.n_mels, 3]));
    out.push(entry(format!("{a}.conv1.bias"), &[e.width]));
    out.push(entry(format!("{a}.conv2.weight"), &[e.width, e.width, 3]));
    out.push(entry(format!("{a}.conv2.bias"), &[e.width]));
    out.push(entry(
        format!("{a}.embed_positions.weight"),
        &[e.positions, e.width],
    ));
    out.push(entry(format!("{a}.layer_norm.weight"), &[e.width]));
    out.push(entry(format!("{a}.layer_norm.bias"), &[e.width]));
    for layer in 0..e.layers {
        let l = format!("{a}.layers.{layer}");
        for (proj, bias) in [
            ("q_proj", true),
            ("k_proj", false),
            ("v_proj", true),
            ("out_proj", true),
        ] {
            out.push(entry(
                format!("{l}.self_attn.{proj}.weight"),
                &[e.width, e.width],
            ));
            if bias {
                out.push(entry(format!("{l}.self_attn.{proj}.bias"), &[e.width]));
            }
        }
        for norm in ["self_attn_layer_norm", "final_layer_norm"] {
            out.push(entry(format!("{l}.{norm}.weight"), &[e.width]));
            out.push(entry(format!("{l}.{norm}.bias"), &[e.width]));
        }
        out.push(entry(format!("{l}.fc1.weight"), &[e.ffn, e.width]));
        out.push(entry(format!("{l}.fc1.bias"), &[e.ffn]));
        out.push(entry(format!("{l}.fc2.weight"), &[e.width, e.ffn]));
        out.push(entry(format!("{l}.fc2.bias"), &[e.width]));
    }
    let p = PROJECTOR_PREFIX;
    out.push(entry(
        format!("{p}.linear_1.weight"),
        &[d.hidden, e.width * config.downsample_factor],
    ));
    out.push(entry(format!("{p}.linear_2.weight"), &[d.hidden, d.hidden]));
    let t = DECODER_PREFIX;
    out.push(entry(EMBEDDING, &[d.vocab, d.hidden]));
    out.push(entry(LM_HEAD, &[d.vocab, d.hidden]));
    out.push(entry(format!("{t}.norm.weight"), &[d.hidden]));
    for (layer, kind) in d.layer_kinds.iter().enumerate() {
        let l = format!("{t}.layers.{layer}");
        let g = d.geometry(layer);
        let q = g.heads * g.head_dim;
        let kv = g.kv_heads * g.head_dim;
        out.push(entry(
            format!("{l}.self_attn.q_proj.weight"),
            &[q, d.hidden],
        ));
        out.push(entry(
            format!("{l}.self_attn.k_proj.weight"),
            &[kv, d.hidden],
        ));
        if *kind == SovLayerKind::Sliding {
            out.push(entry(
                format!("{l}.self_attn.v_proj.weight"),
                &[kv, d.hidden],
            ));
        }
        out.push(entry(
            format!("{l}.self_attn.o_proj.weight"),
            &[d.hidden, q],
        ));
        out.push(entry(format!("{l}.self_attn.q_norm.weight"), &[g.head_dim]));
        out.push(entry(format!("{l}.self_attn.k_norm.weight"), &[g.head_dim]));
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            out.push(entry(format!("{l}.{norm}.weight"), &[d.hidden]));
        }
        out.push(entry(format!("{l}.layer_scalar"), &[1]));
        out.push(entry(
            format!("{l}.mlp.gate_proj.weight"),
            &[d.mlp_width, d.hidden],
        ));
        out.push(entry(
            format!("{l}.mlp.up_proj.weight"),
            &[d.mlp_width, d.hidden],
        ));
        out.push(entry(
            format!("{l}.mlp.down_proj.weight"),
            &[d.hidden, d.mlp_width],
        ));
        out.push(entry(
            format!("{l}.experts.gate_up_proj"),
            &[d.experts, 2 * d.expert_width, d.hidden],
        ));
        out.push(entry(
            format!("{l}.experts.down_proj"),
            &[d.experts, d.hidden, d.expert_width],
        ));
        for (name, shape) in [
            ("router.proj.weight", vec![d.experts, d.hidden]),
            ("router.scale", vec![d.hidden]),
            ("router.per_expert_scale", vec![d.experts]),
        ] {
            out.push(CensusEntry {
                name: format!("{l}.{name}"),
                f32: true,
                shape,
            });
        }
    }
    out.sort();
    out
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

struct Located {
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// The memory-mapped checkpoint.
pub struct SovCheckpoint {
    dir: PathBuf,
    maps: Vec<Mmap>,
    files: Vec<String>,
    tensors: BTreeMap<String, Located>,
}

impl std::fmt::Debug for SovCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovCheckpoint")
            .field("dir", &self.dir)
            .field("files", &self.files.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

fn load_error(message: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("SOV-Transcribe checkpoint: {message}"))
}

impl SovCheckpoint {
    /// Maps every shard the index names and checks the census.
    pub fn open(dir: &Path, config: &SovTranscribeConfig) -> Result<Self> {
        let index_path = dir.join(INDEX_FILE);
        let index: Index = serde_json::from_slice(
            &std::fs::read(&index_path)
                .map_err(|err| load_error(format!("{}: {err}", index_path.display())))?,
        )
        .map_err(|err| load_error(format!("{}: {err}", index_path.display())))?;
        let files = index
            .weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut maps = Vec::with_capacity(files.len());
        let mut tensors = BTreeMap::new();
        for (file_index, file) in files.iter().enumerate() {
            if file.contains('/') || file.contains('\\') || file.contains("..") {
                return Err(load_error(format!("the index names a path, {file:?}")));
            }
            let path = dir.join(file);
            let handle = File::open(&path)
                .map_err(|err| load_error(format!("{}: {err}", path.display())))?;
            // SAFETY: the model directory is not modified while it is loaded.
            let map = unsafe { Mmap::map(&handle) }
                .map_err(|err| load_error(format!("{}: {err}", path.display())))?;
            let (header_len, metadata) = SafeTensors::read_metadata(&map)
                .map_err(|err| load_error(format!("{}: {err}", path.display())))?;
            let data_start = 8 + header_len;
            for (name, info) in metadata.tensors() {
                if index.weight_map.get(&name).map(String::as_str) != Some(file.as_str()) {
                    return Err(load_error(format!(
                        "{file} holds `{name}`, which the index places elsewhere"
                    )));
                }
                let (start, end) = info.data_offsets;
                let (start, end) = (data_start + start, data_start + end);
                if end > map.len() || start > end {
                    return Err(load_error(format!("`{name}` runs past the end of {file}")));
                }
                tensors.insert(
                    name,
                    Located {
                        file: file_index,
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        start,
                        end,
                    },
                );
            }
            maps.push(map);
        }
        if let Some(name) = index
            .weight_map
            .keys()
            .find(|name| !tensors.contains_key(*name))
        {
            return Err(load_error(format!(
                "the index names `{name}`, which no shard holds"
            )));
        }
        let checkpoint = Self {
            dir: dir.to_path_buf(),
            maps,
            files,
            tensors,
        };
        checkpoint.verify_census(config)?;
        Ok(checkpoint)
    }

    fn verify_census(&self, config: &SovTranscribeConfig) -> Result<()> {
        let expected = expected_census(config);
        let mut problems = Vec::new();
        let names = expected
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>();
        for entry in &expected {
            match self.tensors.get(&entry.name) {
                None => problems.push(format!("missing `{}`", entry.name)),
                Some(found) => {
                    let dtype_ok = matches!(
                        (entry.f32, found.dtype),
                        (true, Dtype::F32) | (false, Dtype::BF16) | (false, Dtype::F32)
                    );
                    if !dtype_ok || found.shape != entry.shape {
                        problems.push(format!(
                            "`{}` is {:?} {:?}, expected {} {:?}",
                            entry.name,
                            found.dtype,
                            found.shape,
                            if entry.f32 { "F32" } else { "BF16" },
                            entry.shape
                        ));
                    }
                }
            }
        }
        for name in self.tensors.keys() {
            if !names.contains(name.as_str()) {
                problems.push(format!("unexpected `{name}`"));
            }
        }
        if problems.is_empty() {
            return Ok(());
        }
        let shown = problems
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ");
        Err(load_error(format!(
            "{} tensors differ from the config's geometry: {shown}",
            problems.len()
        )))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn files(&self) -> &[String] {
        &self.files
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Bytes of every tensor the loader uploads (all but the tied head).
    pub fn uploaded_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|(name, _)| name.as_str() != LM_HEAD)
            .map(|(_, located)| (located.end - located.start) as u64)
            .sum()
    }

    fn located(&self, name: &str) -> Result<&Located> {
        self.tensors
            .get(name)
            .ok_or_else(|| load_error(format!("`{name}` is missing")))
    }

    fn bytes(&self, located: &Located) -> &[u8] {
        &self.maps[located.file][located.start..located.end]
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.located(name)?.shape)
    }

    /// Whether the checkpoint stores `name` as BF16.
    pub fn is_bf16(&self, name: &str) -> Result<bool> {
        Ok(self.located(name)?.dtype == Dtype::BF16)
    }

    /// `name` as host F32 values.
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        let located = self.located(name)?;
        let bytes = self.bytes(located);
        match located.dtype {
            Dtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|pair| bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
                .collect()),
            Dtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                .collect()),
            other => Err(load_error(format!("`{name}` has dtype {other:?}"))),
        }
    }

    /// `name` on `device` as `dtype`. A BF16 tensor wanted as BF16 uploads
    /// its mapped bytes as they are (an expert stack never widens to F32 on
    /// the host); anything else converts through host F32.
    pub fn tensor(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let located = self.located(name)?;
        if located.dtype == Dtype::BF16 && dtype == DType::BF16 {
            return Ok(Tensor::from_raw_buffer(
                self.bytes(located),
                DType::BF16,
                &located.shape,
                device,
            )?);
        }
        let values = self.host_f32(name)?;
        Ok(Tensor::from_vec(values, located.shape.as_slice(), device)?.to_dtype(dtype)?)
    }

    /// Whether the stored head equals the embedding byte for byte.
    pub fn head_is_tied(&self) -> Result<bool> {
        let head = self.located(LM_HEAD)?;
        let embedding = self.located(EMBEDDING)?;
        Ok(head.dtype == embedding.dtype
            && head.shape == embedding.shape
            && self.bytes(head) == self.bytes(embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grouped census of the published checkpoint
    /// (`tests/fixtures/sov_transcribe/tensor_census.txt`): `count name
    /// dtype shape`, with layer indices written `N`.
    fn published_groups() -> BTreeMap<(String, String, String), usize> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sov_transcribe/tensor_census.txt");
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .map(|line| {
                let mut parts = line.splitn(4, ' ');
                let count = parts.next().unwrap().parse::<usize>().unwrap();
                let name = parts.next().unwrap().to_string();
                let dtype = parts.next().unwrap().to_string();
                let shape = parts.next().unwrap().to_string();
                ((name, dtype, shape), count)
            })
            .collect()
    }

    #[test]
    fn the_expected_census_is_the_published_checkpoint() {
        let config = super::super::config::tests::published();
        let mut groups: BTreeMap<(String, String, String), usize> = BTreeMap::new();
        let expected = expected_census(&config);
        for entry in &expected {
            let name = entry
                .name
                .split('.')
                .map(|part| {
                    if part.parse::<usize>().is_ok() {
                        "N"
                    } else {
                        part
                    }
                })
                .collect::<Vec<_>>()
                .join(".");
            let dtype = if entry.f32 { "F32" } else { "BF16" }.to_string();
            *groups
                .entry((name, dtype, format!("{:?}", entry.shape)))
                .or_default() += 1;
        }
        assert_eq!(expected.len(), 1147);
        assert_eq!(groups, published_groups());
        // 26.63B parameters, as the Hub reports.
        let parameters = expected
            .iter()
            .map(|entry| entry.shape.iter().product::<usize>() as u64)
            .sum::<u64>();
        assert_eq!(parameters, 26_619_754_270 + 10_901_760);
    }

    #[test]
    fn the_published_checkpoint_has_sixteen_shards() {
        let files = shard_files(PUBLISHED_SHARDS);
        assert_eq!(files[0], "model-00001-of-00016.safetensors");
        assert_eq!(files[15], "model-00016-of-00016.safetensors");
    }
}

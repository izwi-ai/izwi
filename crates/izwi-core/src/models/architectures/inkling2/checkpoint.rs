//! The published checkpoint: memory map, census and tensor access.
//!
//! The census derives the exact tensor set (name, stored dtype, shape) from
//! the config and fails the load on any missing, extra or reshaped tensor
//! before a byte is read. Whether a layer's experts are NVFP4 or MXFP8 is
//! read from the header (NVFP4 projections carry `_amax` tensors); every
//! other choice is fixed.
//!
//! Tensor names are the checkpoint's own (`model.llm.*`, `model.visual.*`).
//! The conventions that differ from what a reader would assume are spelled
//! out where they are used (`trunk`), and each has a test:
//! * `attn_res.attn_queries.{i}` belongs to layer `i + 1`;
//! * the two `middle_cycle_cross_pass_*` tensors hold (query, mlp) and
//!   (key, value) message queries of each loop layer;
//! * `experts.w13_weight` rows interleave gate and up;
//! * `unembed.weight` is padded to the embedding's row count;
//! * `k_sconv.static_w` is indexed by backward distance.
//!
//! `mtp.safetensors` (the multi-token-prediction head) is not read.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;

use super::config::{Inkling2Config, Inkling2LayerKind};
use super::quant::{ExpertFormat, PackedBytes, PackedExperts, MXFP8_BLOCK, NVFP4_BLOCK};
use crate::error::{Error, Result};

pub const INDEX_FILE: &str = "model.safetensors.index.json";
/// The file a checkpoint without an index is read from.
pub const SINGLE_FILE: &str = "model.safetensors";
/// The published weight file.
pub const PUBLISHED_FILE: &str = "model-00001-of-00001.safetensors";
/// The multi-token-prediction head, which this family does not load.
pub const MTP_FILE: &str = "mtp.safetensors";
pub const TEXT_PREFIX: &str = "model.llm";
pub const VISION_PREFIX: &str = "model.visual";

/// The expert projections of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProjection {
    /// Hidden -> interleaved gate and up rows.
    GateUp,
    /// Expert width -> hidden.
    Down,
}

impl ExpertProjection {
    pub fn tensor(self) -> &'static str {
        match self {
            Self::GateUp => "w13_weight",
            Self::Down => "w2_weight",
        }
    }
}

/// How a tensor may be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stored {
    /// BF16 in the published file; F32 is accepted.
    Dense,
    /// F32 in the published file (learned queries, router bias).
    DenseF32,
    U8,
    Fp8,
    F32,
}

impl Stored {
    fn admits(self, dtype: Dtype) -> bool {
        match self {
            Self::Dense => matches!(dtype, Dtype::BF16 | Dtype::F32),
            Self::DenseF32 | Self::F32 => dtype == Dtype::F32,
            Self::U8 => dtype == Dtype::U8,
            Self::Fp8 => dtype == Dtype::F8_E4M3,
        }
    }

    /// The dtype name the published header uses.
    pub fn published_name(self) -> &'static str {
        match self {
            Self::Dense => "BF16",
            Self::DenseF32 | Self::F32 => "F32",
            Self::U8 => "U8",
            Self::Fp8 => "F8_E4M3",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CensusEntry {
    pub name: String,
    pub stored: Stored,
    pub shape: Vec<usize>,
}

fn entry(name: impl Into<String>, stored: Stored, shape: &[usize]) -> CensusEntry {
    CensusEntry {
        name: name.into(),
        stored,
        shape: shape.to_vec(),
    }
}

pub fn layer_prefix(layer: usize) -> String {
    format!("{TEXT_PREFIX}.layers.{layer}")
}

/// The tensors of one expert projection in `format`.
fn expert_entries(
    out: &mut Vec<CensusEntry>,
    name: &str,
    format: ExpertFormat,
    experts: usize,
    rows: usize,
    columns: usize,
) {
    match format {
        ExpertFormat::Nvfp4 => {
            out.push(entry(name, Stored::U8, &[experts, rows, columns / 2]));
            out.push(entry(
                format!("{name}_scale"),
                Stored::Fp8,
                &[experts, rows, columns / NVFP4_BLOCK],
            ));
            out.push(entry(format!("{name}_amax"), Stored::F32, &[experts]));
            out.push(entry(
                format!("{name}_outer_scale_denominator"),
                Stored::F32,
                &[1],
            ));
        }
        ExpertFormat::Mxfp8 => {
            out.push(entry(name, Stored::Fp8, &[experts, rows, columns]));
            out.push(entry(
                format!("{name}_scale"),
                Stored::U8,
                &[experts, rows, columns / MXFP8_BLOCK],
            ));
        }
    }
}

/// Every tensor of a checkpoint with `config`'s geometry whose layer
/// `layer` stores its experts as `formats[layer]`.
pub fn expected_census(config: &Inkling2Config, formats: &[ExpertFormat]) -> Vec<CensusEntry> {
    use Stored::{Dense, DenseF32};
    let c = config;
    let h = c.hidden;
    let t = TEXT_PREFIX;
    let mut out = vec![
        entry(format!("{t}.embed.weight"), Dense, &[c.vocab, h]),
        entry(format!("{t}.embed_norm.weight"), Dense, &[h]),
        entry(format!("{t}.norm.weight"), Dense, &[h]),
        entry(format!("{t}.unembed.weight"), Dense, &[c.vocab, h]),
        entry(format!("{t}.attn_res.output_query"), DenseF32, &[h]),
        entry(
            format!("{t}.middle_cycle_cross_pass_queries"),
            DenseF32,
            &[2 * c.loop_layers(), h],
        ),
        entry(
            format!("{t}.middle_cycle_cross_pass_kv_queries"),
            DenseF32,
            &[2 * c.loop_layers(), h],
        ),
    ];
    let k = c.conv_kernel;
    for layer in 0..c.layers {
        let l = layer_prefix(layer);
        let g = c.heads(layer);
        let full = c.layer_kinds[layer] == Inkling2LayerKind::Full;
        if layer > 0 {
            // `attn_queries.{i}` is the attention-input query of layer
            // `i + 1`; full layers carry query, key and value queries.
            let shape: &[usize] = if full { &[3, h] } else { &[h] };
            out.push(entry(
                format!("{t}.attn_res.attn_queries.{}", layer - 1),
                DenseF32,
                shape,
            ));
        }
        out.push(entry(
            format!("{t}.attn_res.mlp_queries.{layer}"),
            DenseF32,
            &[h],
        ));
        for (name, shape) in [
            ("attn.wq_du.weight", vec![g.query_width(), h]),
            ("attn.wk_dv.weight", vec![g.kv_width(), h]),
            ("attn.wv_dv.weight", vec![g.kv_width(), h]),
            ("attn.wo_ud.weight", vec![h, g.query_width()]),
            ("attn.gate_attn.weight", vec![g.query_width(), h]),
            ("attn.wr_du.weight", vec![g.heads * c.d_rel, h]),
            ("attn.attn_sink_proj.weight", vec![g.heads, h]),
            ("attn.q_l2_clip.scale", vec![g.heads, g.head_dim]),
            ("attn.rel_l2_clip.scale", vec![g.heads, c.d_rel]),
            ("attn.rel_logits_proj.proj", vec![c.d_rel, g.rel_extent]),
            ("attn_sconv.weight", vec![h, 1, k]),
            ("mlp_sconv.weight", vec![h, 1, k]),
            ("mlp_norm.weight", vec![h]),
            ("mlp.gate.weight", vec![c.experts, h]),
        ] {
            out.push(entry(format!("{l}.{name}"), Dense, &shape));
        }
        out.push(entry(format!("{l}.mlp.gate.bias"), DenseF32, &[c.experts]));
        if full {
            for conv in ["k_sconv", "v_sconv"] {
                out.push(entry(
                    format!("{l}.attn.{conv}.static_w"),
                    Dense,
                    &[k, g.kv_width()],
                ));
                out.push(entry(
                    format!("{l}.attn.{conv}.weight_proj"),
                    Dense,
                    &[g.kv_heads * k, h],
                ));
            }
            for norm in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                out.push(entry(format!("{l}.{norm}.weight"), Dense, &[h]));
            }
        } else {
            for conv in ["k_sconv", "v_sconv"] {
                out.push(entry(
                    format!("{l}.attn.{conv}.weight"),
                    Dense,
                    &[g.kv_width(), 1, k],
                ));
            }
            out.push(entry(format!("{l}.attn_norm.weight"), Dense, &[h]));
        }
        if c.is_loop_layer(layer) {
            for which in ["attn", "k", "v", "mlp"] {
                let p = format!("{l}.controllers.{which}");
                out.push(entry(
                    format!("{p}.down_weight"),
                    Dense,
                    &[c.loop_gate_rank, 2 * h],
                ));
                out.push(entry(
                    format!("{p}.forget_weight"),
                    Dense,
                    &[h, c.loop_gate_rank],
                ));
                out.push(entry(
                    format!("{p}.gain_weight"),
                    Dense,
                    &[h, c.loop_gate_rank],
                ));
            }
        }
        expert_entries(
            &mut out,
            &format!("{l}.mlp.experts.w13_weight"),
            formats[layer],
            c.experts,
            c.gate_up_rows(),
            h,
        );
        expert_entries(
            &mut out,
            &format!("{l}.mlp.experts.w2_weight"),
            formats[layer],
            c.experts,
            h,
            c.expert_width,
        );
    }
    if let Some(v) = &c.vision {
        let b = format!("{VISION_PREFIX}.backbone");
        let w = v.hidden;
        out.push(entry(
            format!("{b}.embeddings.class_embedding"),
            Dense,
            &[w],
        ));
        out.push(entry(
            format!("{b}.embeddings.cls_position_embedding"),
            Dense,
            &[w],
        ));
        out.push(entry(
            format!("{b}.embeddings.patch_embedding.weight"),
            Dense,
            &[w, 3 * v.patch * v.patch],
        ));
        out.push(entry(
            format!("{b}.embeddings.position_embedding"),
            Dense,
            &[v.grid * v.grid, w],
        ));
        out.push(entry(format!("{b}.ln_pre.weight"), Dense, &[w]));
        out.push(entry(format!("{b}.ln_pre.bias"), Dense, &[w]));
        for layer in 0..v.layers {
            let l = format!("{b}.layers.{layer}");
            for norm in ["ln_1", "ln_2"] {
                out.push(entry(format!("{l}.{norm}.weight"), Dense, &[w]));
                out.push(entry(format!("{l}.{norm}.bias"), Dense, &[w]));
            }
            out.push(entry(format!("{l}.ls_1.gamma"), Dense, &[w]));
            out.push(entry(format!("{l}.ls_2.gamma"), Dense, &[w]));
            out.push(entry(
                format!("{l}.mlp.c_fc.weight"),
                Dense,
                &[v.mlp_width, w],
            ));
            out.push(entry(format!("{l}.mlp.c_fc.bias"), Dense, &[v.mlp_width]));
            out.push(entry(
                format!("{l}.mlp.c_proj.weight"),
                Dense,
                &[w, v.mlp_width],
            ));
            out.push(entry(format!("{l}.mlp.c_proj.bias"), Dense, &[w]));
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                out.push(entry(
                    format!("{l}.self_attn.{proj}.weight"),
                    Dense,
                    &[w, w],
                ));
                out.push(entry(format!("{l}.self_attn.{proj}.bias"), Dense, &[w]));
            }
        }
        let merged = v.merge * v.merge * w;
        out.push(entry(
            format!("{VISION_PREFIX}.proj_in.weight"),
            Dense,
            &[v.projection_hidden, merged],
        ));
        out.push(entry(
            format!("{VISION_PREFIX}.proj_out.weight"),
            Dense,
            &[c.hidden, v.projection_hidden],
        ));
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
pub struct Inkling2Checkpoint {
    dir: PathBuf,
    files: Vec<String>,
    maps: Vec<Arc<Mmap>>,
    tensors: BTreeMap<String, Located>,
    formats: Vec<ExpertFormat>,
}

impl std::fmt::Debug for Inkling2Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inkling2Checkpoint")
            .field("dir", &self.dir)
            .field("files", &self.files)
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

fn load_error(message: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Inkling 2 checkpoint: {message}"))
}

/// The weight files of a model directory: the index's, or the single file.
pub fn weight_files(dir: &Path) -> Result<Vec<String>> {
    let index_path = dir.join(INDEX_FILE);
    if !index_path.exists() {
        return Ok(vec![SINGLE_FILE.to_string()]);
    }
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
    if files.is_empty() {
        return Err(load_error("the index names no weight file"));
    }
    if let Some(file) = files
        .iter()
        .find(|file| file.contains('/') || file.contains('\\') || file.contains(".."))
    {
        return Err(load_error(format!("the index names a path, {file:?}")));
    }
    Ok(files)
}

impl Inkling2Checkpoint {
    /// Maps the weight files and checks the census against `config`.
    pub fn open(dir: &Path, config: &Inkling2Config) -> Result<Self> {
        let files = weight_files(dir)?;
        let mut maps = Vec::with_capacity(files.len());
        let mut tensors = BTreeMap::new();
        for (file_index, file) in files.iter().enumerate() {
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
                let (start, end) = info.data_offsets;
                let (start, end) = (data_start + start, data_start + end);
                if end > map.len() || start > end {
                    return Err(load_error(format!("`{name}` runs past the end of {file}")));
                }
                let located = Located {
                    file: file_index,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    start,
                    end,
                };
                if tensors.insert(name.clone(), located).is_some() {
                    return Err(load_error(format!("`{name}` is stored twice")));
                }
            }
            maps.push(Arc::new(map));
        }
        let formats = (0..config.layers)
            .map(|layer| {
                let amax = format!("{}.mlp.experts.w13_weight_amax", layer_prefix(layer));
                if tensors.contains_key(&amax) {
                    ExpertFormat::Nvfp4
                } else {
                    ExpertFormat::Mxfp8
                }
            })
            .collect::<Vec<_>>();
        let checkpoint = Self {
            dir: dir.to_path_buf(),
            files,
            maps,
            tensors,
            formats,
        };
        checkpoint.verify_census(config)?;
        Ok(checkpoint)
    }

    fn verify_census(&self, config: &Inkling2Config) -> Result<()> {
        let expected = expected_census(config, &self.formats);
        let names = expected
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>();
        let mut problems = Vec::new();
        for entry in &expected {
            match self.tensors.get(&entry.name) {
                None => problems.push(format!("missing `{}`", entry.name)),
                Some(found) => {
                    if !entry.stored.admits(found.dtype) || found.shape != entry.shape {
                        problems.push(format!(
                            "`{}` is {:?} {:?}, expected {} {:?}",
                            entry.name,
                            found.dtype,
                            found.shape,
                            entry.stored.published_name(),
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

    /// How layer `layer` stores its experts.
    pub fn expert_format(&self, layer: usize) -> ExpertFormat {
        self.formats[layer]
    }

    pub fn expert_formats(&self) -> &[ExpertFormat] {
        &self.formats
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn located(&self, name: &str) -> Result<&Located> {
        self.tensors
            .get(name)
            .ok_or_else(|| load_error(format!("`{name}` is missing")))
    }

    fn bytes(&self, located: &Located) -> &[u8] {
        &self.maps[located.file][located.start..located.end]
    }

    fn mapped(&self, name: &str) -> Result<PackedBytes> {
        let located = self.located(name)?;
        Ok(PackedBytes::Mapped {
            map: self.maps[located.file].clone(),
            start: located.start,
            len: located.end - located.start,
        })
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.located(name)?.shape)
    }

    /// Stored bytes of `name`.
    pub fn stored_bytes(&self, name: &str) -> Result<u64> {
        let located = self.located(name)?;
        Ok((located.end - located.start) as u64)
    }

    /// Stored bytes of every tensor whose name starts with `prefix`.
    pub fn stored_bytes_under(&self, prefix: &str) -> u64 {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(_, located)| (located.end - located.start) as u64)
            .sum()
    }

    /// `name` as host F32 values (BF16 or F32 tensors).
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

    /// The first `rows` rows of the 2-D tensor `name` as host F32.
    pub fn host_f32_rows(&self, name: &str, rows: usize) -> Result<Vec<f32>> {
        let located = self.located(name)?;
        if located.shape.len() != 2 || rows > located.shape[0] {
            return Err(load_error(format!("`{name}` has no {rows} leading rows")));
        }
        let mut values = self.host_f32(name)?;
        values.truncate(rows * located.shape[1]);
        Ok(values)
    }

    /// `name` on `device` as `dtype`. A BF16 tensor wanted as BF16 uploads
    /// its mapped bytes as they are; anything else converts through host
    /// F32.
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

    /// The first `rows` rows of the 2-D tensor `name` on `device`.
    pub fn tensor_rows(
        &self,
        name: &str,
        rows: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let located = self.located(name)?;
        if located.shape.len() != 2 || rows > located.shape[0] {
            return Err(load_error(format!("`{name}` has no {rows} leading rows")));
        }
        let columns = located.shape[1];
        if located.dtype == Dtype::BF16 && dtype == DType::BF16 {
            return Ok(Tensor::from_raw_buffer(
                &self.bytes(located)[..rows * columns * 2],
                DType::BF16,
                &[rows, columns],
                device,
            )?);
        }
        let values = self.host_f32_rows(name, rows)?;
        Ok(Tensor::from_vec(values, (rows, columns), device)?.to_dtype(dtype)?)
    }

    /// One expert projection of `layer`, still packed, backed by the map.
    pub fn experts(&self, layer: usize, projection: ExpertProjection) -> Result<PackedExperts> {
        let name = format!(
            "{}.mlp.experts.{}",
            layer_prefix(layer),
            projection.tensor()
        );
        let scale = format!("{name}_scale");
        let shape = self.shape(&name)?;
        let (experts, rows) = (shape[0], shape[1]);
        match self.formats[layer] {
            ExpertFormat::Nvfp4 => {
                let columns = shape[2] * 2;
                let denominator = self.host_f32(&format!("{name}_outer_scale_denominator"))?[0];
                PackedExperts::nvfp4(
                    experts,
                    rows,
                    columns,
                    self.mapped(&name)?,
                    self.mapped(&scale)?,
                    &self.host_f32(&format!("{name}_amax"))?,
                    denominator,
                )
            }
            ExpertFormat::Mxfp8 => PackedExperts::mxfp8(
                experts,
                rows,
                shape[2],
                self.mapped(&name)?,
                self.mapped(&scale)?,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{fixture, published, tiny};
    use super::*;

    /// `file name dtype [shape]` lines of the published header
    /// (`tests/fixtures/inkling2/tensor_census.txt`).
    fn published_census(file: &str) -> BTreeSet<(String, String, String)> {
        std::fs::read_to_string(fixture("inkling2").join("tensor_census.txt"))
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .filter_map(|line| {
                let mut parts = line.splitn(4, ' ');
                let found = parts.next().unwrap();
                let name = parts.next().unwrap().to_string();
                let dtype = parts.next().unwrap().to_string();
                let shape = parts.next().unwrap().to_string();
                (found == file).then_some((name, dtype, shape))
            })
            .collect()
    }

    /// As published: NVFP4 experts on layers 0..20, MXFP8 on the last four.
    fn published_formats() -> Vec<ExpertFormat> {
        (0..24)
            .map(|layer| {
                if layer < 20 {
                    ExpertFormat::Nvfp4
                } else {
                    ExpertFormat::Mxfp8
                }
            })
            .collect()
    }

    #[test]
    fn the_expected_census_is_the_published_checkpoint() {
        let config = published();
        let expected = expected_census(&config, &published_formats())
            .into_iter()
            .map(|entry| {
                let shape = entry
                    .shape
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                (
                    entry.name,
                    entry.stored.published_name().to_string(),
                    format!("[{shape}]"),
                )
            })
            .collect::<BTreeSet<_>>();
        let published = published_census(PUBLISHED_FILE);
        assert_eq!(published.len(), 1244);
        let missing = published.difference(&expected).take(5).collect::<Vec<_>>();
        let extra = expected.difference(&published).take(5).collect::<Vec<_>>();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "published but not expected: {missing:?}; expected but not published: {extra:?}"
        );
        // The head file is a separate 54-tensor census this family does not read.
        assert_eq!(published_census(MTP_FILE).len(), 54);
    }

    #[test]
    fn the_tiny_checkpoint_opens_and_reports_both_expert_formats() {
        let dir = fixture("inkling2_golden");
        let config = tiny();
        let checkpoint = Inkling2Checkpoint::open(&dir, &config).unwrap();
        assert_eq!(checkpoint.files(), [SINGLE_FILE.to_string()]);
        assert_eq!(checkpoint.expert_format(0), ExpertFormat::Nvfp4);
        assert_eq!(checkpoint.expert_format(5), ExpertFormat::Nvfp4);
        assert_eq!(checkpoint.expert_format(7), ExpertFormat::Mxfp8);
        let gate_up = checkpoint.experts(0, ExpertProjection::GateUp).unwrap();
        assert_eq!(
            (gate_up.experts(), gate_up.rows(), gate_up.columns()),
            (config.experts, config.gate_up_rows(), config.hidden)
        );
        let down = checkpoint.experts(7, ExpertProjection::Down).unwrap();
        assert_eq!(
            (down.rows(), down.columns()),
            (config.hidden, config.expert_width)
        );
        assert_eq!(down.format(), ExpertFormat::Mxfp8);
        let rows = checkpoint
            .host_f32_rows("model.llm.unembed.weight", config.unpadded_vocab)
            .unwrap();
        assert_eq!(rows.len(), config.unpadded_vocab * config.hidden);
    }

    #[test]
    fn a_checkpoint_with_another_geometry_is_refused() {
        let dir = fixture("inkling2_golden");
        let mut config = tiny();
        config.loop_gate_rank += 1;
        let error = Inkling2Checkpoint::open(&dir, &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("controllers"), "{error}");
        let mut config = tiny();
        config.layers -= 1;
        config.layer_kinds.pop();
        let error = Inkling2Checkpoint::open(&dir, &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unexpected"), "{error}");
    }
}
